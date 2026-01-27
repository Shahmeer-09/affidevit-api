"""
Dashboard service for confidence metrics and learning loop analysis.
Provides data for the admin confidence dashboard.
"""

from datetime import timedelta
from decimal import Decimal
from django.db.models import Count, Avg, Q, F
from django.db.models.functions import TruncDate
from django.utils import timezone


def get_dashboard_data():
    """
    Get full confidence dashboard data for all affidavit types.
    
    Returns:
        dict with summary stats and per-type metrics
    """
    from affidavits.models import AffidavitType, Request, AIRun
    
    now = timezone.now()
    thirty_days_ago = now - timedelta(days=30)
    
    # Get all active affidavit types with metrics
    types_data = []
    
    for aff_type in AffidavitType.objects.filter(enabled_on_homepage=True):
        type_metrics = calculate_type_metrics(aff_type, thirty_days_ago)
        types_data.append(type_metrics)
    
    # Calculate overall stats
    total_requests = Request.objects.filter(created_at__gte=thirty_days_ago).count()
    approved_requests = Request.objects.filter(
        created_at__gte=thirty_days_ago,
        status=Request.Status.APPROVED
    ).count()
    
    avg_latency = AIRun.objects.filter(
        created_at__gte=thirty_days_ago
    ).aggregate(avg=Avg('latency_ms'))['avg'] or 0
    
    total_overrides = Request.objects.filter(
        created_at__gte=thirty_days_ago,
        qa_overridden=True
    ).count()
    
    return {
        'summary': {
            'total_requests_30d': total_requests,
            'approval_rate': round((approved_requests / total_requests * 100) if total_requests > 0 else 0, 1),
            'avg_latency_ms': round(avg_latency),
            'total_overrides_30d': total_overrides,
            'types_in_instant_mode': sum(1 for t in types_data if t['default_mode'] == 'instant'),
            'types_pending_promotion': sum(1 for t in types_data if t['promotion_ready']),
        },
        'types': sorted(types_data, key=lambda x: -x['volume_30d']),
        'generated_at': now.isoformat()
    }


def calculate_type_metrics(affidavit_type, since_date):
    """
    Calculate confidence metrics for a single affidavit type.
    
    Args:
        affidavit_type: AffidavitType instance
        since_date: datetime to calculate metrics from
        
    Returns:
        dict with all metrics for this type
    """
    from affidavits.models import Request, AIRun, ReviewerEdit
    
    # Base queryset for this type
    requests = Request.objects.filter(
        affidavit_type=affidavit_type,
        created_at__gte=since_date
    )
    
    volume = requests.count()
    
    # QA pass rate (no issues flagged)
    qa_passed = requests.filter(qa_passed=True).count()
    qa_pass_rate = (qa_passed / volume * 100) if volume > 0 else 0
    
    # Override rate
    overrides = requests.filter(qa_overridden=True).count()
    override_rate = (overrides / volume * 100) if volume > 0 else 0
    
    # Edit rate (significant edits by reviewers)
    edited = requests.filter(draft_edited_significantly=True).count()
    edit_rate = (edited / volume * 100) if volume > 0 else 0
    
    # Rejection rate
    rejected = requests.filter(status=Request.Status.REJECTED).count()
    rejection_rate = (rejected / volume * 100) if volume > 0 else 0
    
    # Friction reports
    friction_count = requests.filter(friction_reports__isnull=False).distinct().count()
    friction_rate = (friction_count / volume * 100) if volume > 0 else 0
    
    # Average processing time
    completed = requests.filter(
        status__in=[Request.Status.COMPLETED, Request.Status.APPROVED]
    ).exclude(time_to_complete_seconds=0)
    avg_time = completed.aggregate(avg=Avg('time_to_complete_seconds'))['avg'] or 0
    
    # AI metrics
    ai_runs = AIRun.objects.filter(request__in=requests)
    avg_latency = ai_runs.aggregate(avg=Avg('latency_ms'))['avg'] or 0
    total_tokens = ai_runs.aggregate(
        total=Avg('total_tokens')
    )['total'] or 0
    total_cost = ai_runs.aggregate(
        cost=Avg('estimated_cost_usd')
    )['cost'] or 0
    
    # Scenario distribution
    scenarios = {}
    for req in requests.exclude(scenario_tags=[]).values_list('scenario_tags', flat=True):
        for tag in (req or []):
            scenarios[tag] = scenarios.get(tag, 0) + 1
    
    # Calculate confidence score (0-100)
    confidence_score = calculate_confidence_score(
        volume=volume,
        qa_pass_rate=qa_pass_rate,
        override_rate=override_rate,
        edit_rate=edit_rate,
        rejection_rate=rejection_rate,
        tier_threshold=affidavit_type.min_volume_threshold
    )
    
    # Check if ready for promotion
    promotion_ready = check_promotion_eligibility(
        affidavit_type=affidavit_type,
        volume=volume,
        confidence_score=confidence_score,
        override_rate=override_rate,
        rejection_rate=rejection_rate
    )
    
    return {
        'id': affidavit_type.id,
        'name': affidavit_type.name,
        'tier': affidavit_type.tier,
        'default_mode': affidavit_type.default_mode,
        'confidence_status': affidavit_type.confidence_status,
        'volume_30d': volume,
        'min_volume_threshold': affidavit_type.min_volume_threshold,
        'metrics': {
            'qa_pass_rate': round(qa_pass_rate, 1),
            'override_rate': round(override_rate, 1),
            'edit_rate': round(edit_rate, 1),
            'rejection_rate': round(rejection_rate, 1),
            'friction_rate': round(friction_rate, 1),
            'avg_time_seconds': round(avg_time),
            'avg_latency_ms': round(avg_latency),
            'avg_tokens': round(total_tokens),
            'avg_cost_usd': round(float(total_cost), 4),
        },
        'scenarios': dict(sorted(scenarios.items(), key=lambda x: -x[1])[:10]),
        'confidence_score': round(confidence_score, 1),
        'promotion_ready': promotion_ready,
        'versions': {
            'policy': affidavit_type.policy_version,
            'prompt': affidavit_type.prompt_pack_version,
            'template': affidavit_type.template_version,
        }
    }


def calculate_confidence_score(
    volume: int,
    qa_pass_rate: float,
    override_rate: float,
    edit_rate: float,
    rejection_rate: float,
    tier_threshold: int
) -> float:
    """
    Calculate a 0-100 confidence score based on metrics.
    
    Weights:
    - Volume adequacy: 20%
    - QA pass rate: 40%
    - Low override rate: 15%
    - Low edit rate: 15%
    - Low rejection rate: 10%
    """
    # Volume score (0-100 based on meeting threshold)
    volume_score = min(100, (volume / tier_threshold) * 100) if tier_threshold > 0 else 0
    
    # QA pass rate is already 0-100
    qa_score = qa_pass_rate
    
    # Override score: lower is better (invert)
    override_score = max(0, 100 - override_rate * 5)  # 20% override = 0 score
    
    # Edit score: lower is better (invert)
    edit_score = max(0, 100 - edit_rate * 2)  # 50% edit = 0 score
    
    # Rejection score: lower is better (invert)
    rejection_score = max(0, 100 - rejection_rate * 10)  # 10% rejection = 0 score
    
    # Weighted average
    confidence = (
        volume_score * 0.20 +
        qa_score * 0.40 +
        override_score * 0.15 +
        edit_score * 0.15 +
        rejection_score * 0.10
    )
    
    return confidence


def check_promotion_eligibility(
    affidavit_type,
    volume: int,
    confidence_score: float,
    override_rate: float,
    rejection_rate: float
) -> bool:
    """
    Check if an affidavit type is ready for promotion to instant mode.
    
    Criteria:
    - Volume meets tier threshold
    - Confidence score >= 85
    - Override rate < 5%
    - Rejection rate < 2%
    - Currently in 'review' mode
    """
    if affidavit_type.default_mode == 'instant':
        return False  # Already in instant mode
        
    if volume < affidavit_type.min_volume_threshold:
        return False
        
    if confidence_score < 85:
        return False
        
    if override_rate >= 5:
        return False
        
    if rejection_rate >= 2:
        return False
    
    return True


def update_all_confidence_metrics():
    """
    Update confidence status for all affidavit types based on current metrics.
    Called by periodic Celery task.
    """
    from affidavits.models import AffidavitType
    
    now = timezone.now()
    thirty_days_ago = now - timedelta(days=30)
    
    updated = []
    
    for aff_type in AffidavitType.objects.all():
        metrics = calculate_type_metrics(aff_type, thirty_days_ago)
        
        old_status = aff_type.confidence_status
        
        # Update confidence status based on score
        score = metrics['confidence_score']
        if score >= 85:
            new_status = AffidavitType.ConfidenceStatus.CONFIDENT
        elif score >= 60:
            new_status = AffidavitType.ConfidenceStatus.CONTROLLED
        else:
            new_status = AffidavitType.ConfidenceStatus.LEARNING
        
        if old_status != new_status:
            aff_type.confidence_status = new_status
            aff_type.save(update_fields=['confidence_status', 'updated_at'])
            updated.append({
                'id': aff_type.id,
                'name': aff_type.name,
                'old_status': old_status,
                'new_status': new_status,
                'score': score
            })
    
    return {
        'success': True,
        'updated_count': len(updated),
        'updated_types': updated
    }


def get_weekly_learning_report():
    """
    Generate a weekly learning report for AI improvement.
    Analyzes overrides, edits, and scenarios to identify patterns.
    """
    from affidavits.models import AffidavitType, Request, ReviewerEdit
    
    now = timezone.now()
    week_ago = now - timedelta(days=7)
    
    report = {
        'period': {
            'start': week_ago.isoformat(),
            'end': now.isoformat()
        },
        'types': [],
        'top_override_reasons': [],
        'new_scenarios': [],
        'recommendations': []
    }
    
    # Analyze each active type
    for aff_type in AffidavitType.objects.filter(enabled_on_homepage=True):
        requests = Request.objects.filter(
            affidavit_type=aff_type,
            created_at__gte=week_ago
        )
        
        volume = requests.count()
        if volume == 0:
            continue
        
        overrides = requests.filter(qa_overridden=True)
        edits = ReviewerEdit.objects.filter(
            request__in=requests
        )
        
        # Collect override reasons
        override_notes = list(overrides.exclude(
            override_notes=''
        ).values_list('override_notes', flat=True))
        
        # Collect edit patterns
        edit_patterns = {}
        for edit in edits.values('issue_type'):
            issue = edit['issue_type']
            edit_patterns[issue] = edit_patterns.get(issue, 0) + 1
        
        # New scenarios detected
        new_scenarios = list(requests.filter(
            new_scenario_flag=True
        ).values_list('scenario_tags', flat=True))
        
        type_data = {
            'id': aff_type.id,
            'name': aff_type.name,
            'volume': volume,
            'overrides': overrides.count(),
            'override_rate': round(overrides.count() / volume * 100, 1),
            'edit_patterns': dict(sorted(edit_patterns.items(), key=lambda x: -x[1])[:5]),
            'new_scenarios_count': len(new_scenarios)
        }
        
        report['types'].append(type_data)
        report['top_override_reasons'].extend(override_notes)
        
        for scenario_list in new_scenarios:
            if scenario_list:
                report['new_scenarios'].extend(scenario_list)
    
    # Deduplicate and count
    report['new_scenarios'] = list(set(report['new_scenarios']))[:20]
    
    # Count override reasons
    reason_counts = {}
    for reason in report['top_override_reasons']:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report['top_override_reasons'] = [
        {'reason': r, 'count': c}
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]
    ]
    
    # Generate recommendations
    for type_data in report['types']:
        if type_data['override_rate'] > 10:
            report['recommendations'].append({
                'type': 'high_override_rate',
                'affidavit_type': type_data['name'],
                'message': f"High override rate ({type_data['override_rate']}%). Review QA prompts for {type_data['name']}."
            })
        
        if type_data['new_scenarios_count'] > 5:
            report['recommendations'].append({
                'type': 'new_scenarios',
                'affidavit_type': type_data['name'],
                'message': f"Multiple new scenarios detected for {type_data['name']}. Consider updating scenario library."
            })
    
    return report


def get_type_trend_data(affidavit_type_id: int, days: int = 30):
    """
    Get daily trend data for a specific affidavit type.
    
    Returns:
        dict with daily volume, pass rate, and other metrics
    """
    from affidavits.models import AffidavitType, Request
    
    try:
        aff_type = AffidavitType.objects.get(id=affidavit_type_id)
    except AffidavitType.DoesNotExist:
        return {'error': 'Affidavit type not found'}
    
    now = timezone.now()
    start_date = now - timedelta(days=days)
    
    daily_data = (
        Request.objects
        .filter(affidavit_type=aff_type, created_at__gte=start_date)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            total=Count('id'),
            qa_passed=Count('id', filter=Q(qa_passed=True)),
            overrides=Count('id', filter=Q(qa_overridden=True)),
            completed=Count('id', filter=Q(status=Request.Status.COMPLETED))
        )
        .order_by('date')
    )
    
    trend = []
    for day in daily_data:
        total = day['total']
        trend.append({
            'date': day['date'].isoformat(),
            'volume': total,
            'qa_pass_rate': round(day['qa_passed'] / total * 100, 1) if total > 0 else 0,
            'override_rate': round(day['overrides'] / total * 100, 1) if total > 0 else 0,
            'completion_rate': round(day['completed'] / total * 100, 1) if total > 0 else 0
        })
    
    return {
        'affidavit_type': {
            'id': aff_type.id,
            'name': aff_type.name
        },
        'period_days': days,
        'daily_trend': trend
    }


def get_cost_dashboard(days: int = 30):
    """
    Get AI cost analytics dashboard data.
    
    Returns:
        dict with cost breakdown by type, model, and time period
    """
    from affidavits.models import AffidavitType, AIRun
    from django.db.models import Sum, Avg, Count
    
    now = timezone.now()
    start_date = now - timedelta(days=days)
    
    # Overall stats
    overall = AIRun.objects.filter(created_at__gte=start_date).aggregate(
        total_cost=Sum('estimated_cost_usd'),
        total_tokens=Sum('total_tokens'),
        total_runs=Count('id'),
        avg_latency=Avg('latency_ms')
    )
    
    # Cost by model
    by_model = (
        AIRun.objects
        .filter(created_at__gte=start_date)
        .values('model_name')
        .annotate(
            cost=Sum('estimated_cost_usd'),
            tokens=Sum('total_tokens'),
            runs=Count('id'),
            avg_latency=Avg('latency_ms')
        )
        .order_by('-cost')
    )
    
    # Cost by node type
    by_node = (
        AIRun.objects
        .filter(created_at__gte=start_date)
        .values('node_type')
        .annotate(
            cost=Sum('estimated_cost_usd'),
            tokens=Sum('total_tokens'),
            runs=Count('id'),
            success_rate=Avg('status')  # Will calculate properly below
        )
        .order_by('-cost')
    )
    
    # Recalculate success rate properly
    node_data = []
    for node in by_node:
        node_type = node['node_type']
        total = AIRun.objects.filter(
            created_at__gte=start_date,
            node_type=node_type
        ).count()
        success = AIRun.objects.filter(
            created_at__gte=start_date,
            node_type=node_type,
            status='success'
        ).count()
        
        node_data.append({
            'node_type': node_type,
            'cost': float(node['cost'] or 0),
            'tokens': node['tokens'] or 0,
            'runs': node['runs'],
            'success_rate': round(success / total * 100, 1) if total > 0 else 0
        })
    
    # Cost by affidavit type
    by_type = (
        AIRun.objects
        .filter(created_at__gte=start_date)
        .values('request__affidavit_type__id', 'request__affidavit_type__name')
        .annotate(
            cost=Sum('estimated_cost_usd'),
            tokens=Sum('total_tokens'),
            runs=Count('id'),
            avg_latency=Avg('latency_ms')
        )
        .order_by('-cost')
    )
    
    type_data = [{
        'id': t['request__affidavit_type__id'],
        'name': t['request__affidavit_type__name'],
        'cost': float(t['cost'] or 0),
        'tokens': t['tokens'] or 0,
        'runs': t['runs'],
        'avg_latency': round(t['avg_latency'] or 0),
        'cost_per_request': round(float(t['cost'] or 0) / t['runs'], 4) if t['runs'] > 0 else 0
    } for t in by_type if t['request__affidavit_type__id']]
    
    # Daily cost trend
    daily_costs = (
        AIRun.objects
        .filter(created_at__gte=start_date)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            cost=Sum('estimated_cost_usd'),
            tokens=Sum('total_tokens'),
            runs=Count('id')
        )
        .order_by('date')
    )
    
    daily_trend = [{
        'date': d['date'].isoformat(),
        'cost': float(d['cost'] or 0),
        'tokens': d['tokens'] or 0,
        'runs': d['runs']
    } for d in daily_costs]
    
    # Projected monthly cost
    if days >= 7 and daily_trend:
        avg_daily_cost = float(overall['total_cost'] or 0) / days
        projected_monthly = avg_daily_cost * 30
    else:
        projected_monthly = 0
    
    return {
        'period_days': days,
        'generated_at': now.isoformat(),
        'summary': {
            'total_cost_usd': float(overall['total_cost'] or 0),
            'total_tokens': overall['total_tokens'] or 0,
            'total_runs': overall['total_runs'] or 0,
            'avg_latency_ms': round(overall['avg_latency'] or 0),
            'avg_cost_per_run': round(float(overall['total_cost'] or 0) / (overall['total_runs'] or 1), 4),
            'projected_monthly_cost': round(projected_monthly, 2)
        },
        'by_model': [{
            'model': m['model_name'],
            'cost': float(m['cost'] or 0),
            'tokens': m['tokens'] or 0,
            'runs': m['runs'],
            'avg_latency': round(m['avg_latency'] or 0)
        } for m in by_model],
        'by_node_type': node_data,
        'by_affidavit_type': type_data,
        'daily_trend': daily_trend
    }


def generate_learning_suggestions(days: int = 7):
    """
    Generate automated learning suggestions based on recent patterns.
    Analyzes overrides, edits, and new scenarios to provide actionable recommendations.
    
    Returns:
        dict with prioritized suggestions for improving AI behavior
    """
    from affidavits.models import AffidavitType, Request, ReviewerEdit, FrictionReport
    
    now = timezone.now()
    start_date = now - timedelta(days=days)
    
    suggestions = []
    
    # Analyze each affidavit type
    for aff_type in AffidavitType.objects.filter(is_active=True):
        requests = Request.objects.filter(
            affidavit_type=aff_type,
            created_at__gte=start_date
        )
        
        total = requests.count()
        if total < 5:  # Skip types with low volume
            continue
        
        # 1. High override rate analysis
        overrides = requests.filter(qa_overridden=True)
        override_rate = (overrides.count() / total) * 100 if total > 0 else 0
        
        if override_rate > 10:
            override_notes = list(overrides.exclude(
                override_notes=''
            ).values_list('override_notes', flat=True)[:5])
            
            suggestions.append({
                'priority': 'high',
                'type': 'qa_tuning',
                'affidavit_type': aff_type.name,
                'affidavit_type_id': aff_type.id,
                'metric': f'{round(override_rate, 1)}% override rate',
                'issue': 'QA is flagging too many false positives',
                'suggestion': 'Review and relax QA validation rules in policy_json',
                'evidence': override_notes,
                'action': {
                    'type': 'update_policy',
                    'field': 'validation_rules',
                    'recommendation': 'Remove or adjust overly strict validation rules'
                }
            })
        
        # 2. High edit rate analysis
        edits = ReviewerEdit.objects.filter(
            request__in=requests
        )
        edit_rate = (requests.filter(draft_edited_significantly=True).count() / total) * 100 if total > 0 else 0
        
        if edit_rate > 15:
            # Group edits by issue type
            edit_types = {}
            for edit in edits.values('issue_type'):
                t = edit['issue_type']
                edit_types[t] = edit_types.get(t, 0) + 1
            
            top_issue = max(edit_types.items(), key=lambda x: x[1]) if edit_types else ('unknown', 0)
            
            suggestions.append({
                'priority': 'high',
                'type': 'draft_improvement',
                'affidavit_type': aff_type.name,
                'affidavit_type_id': aff_type.id,
                'metric': f'{round(edit_rate, 1)}% significant edit rate',
                'issue': f'Most common edit type: {top_issue[0]}',
                'suggestion': 'Add few-shot examples addressing this issue type',
                'evidence': dict(sorted(edit_types.items(), key=lambda x: -x[1])[:3]),
                'action': {
                    'type': 'add_examples',
                    'field': 'few_shot_examples',
                    'recommendation': f'Add 2-3 examples showing correct handling of {top_issue[0]} issues'
                }
            })
        
        # 3. New scenario detection
        new_scenario_count = requests.filter(new_scenario_flag=True).count()
        new_scenario_rate = (new_scenario_count / total) * 100 if total > 0 else 0
        
        if new_scenario_rate > 20:
            # Get sample scenario tags from these requests
            sample_tags = []
            for req in requests.filter(new_scenario_flag=True)[:5]:
                if req.scenario_tags:
                    sample_tags.extend(req.scenario_tags)
            
            suggestions.append({
                'priority': 'medium',
                'type': 'scenario_expansion',
                'affidavit_type': aff_type.name,
                'affidavit_type_id': aff_type.id,
                'metric': f'{round(new_scenario_rate, 1)}% unrecognized scenarios',
                'issue': 'Many requests don\'t match known scenario patterns',
                'suggestion': 'Expand scenario_library with new patterns',
                'evidence': list(set(sample_tags))[:10],
                'action': {
                    'type': 'update_scenarios',
                    'field': 'scenario_library',
                    'recommendation': 'Add new scenario entries based on common answer patterns'
                }
            })
        
        # 4. Friction report analysis
        friction = FrictionReport.objects.filter(
            request__in=requests,
            is_resolved=False
        )
        friction_count = friction.count()
        
        if friction_count >= 3:
            friction_reasons = list(friction.values_list('reason', flat=True)[:5])
            
            suggestions.append({
                'priority': 'high',
                'type': 'commissioner_friction',
                'affidavit_type': aff_type.name,
                'affidavit_type_id': aff_type.id,
                'metric': f'{friction_count} unresolved friction reports',
                'issue': 'Commissioners are refusing to stamp documents',
                'suggestion': 'Review and fix document formatting or wording issues',
                'evidence': friction_reasons,
                'action': {
                    'type': 'review_template',
                    'field': 'template',
                    'recommendation': 'Check template for formatting issues and update disallowed_phrases'
                }
            })
        
        # 5. Low QA pass rate
        qa_pass_rate = (requests.filter(qa_passed=True).count() / total) * 100 if total > 0 else 0
        
        if qa_pass_rate < 50:
            suggestions.append({
                'priority': 'medium',
                'type': 'intake_improvement',
                'affidavit_type': aff_type.name,
                'affidavit_type_id': aff_type.id,
                'metric': f'{round(qa_pass_rate, 1)}% QA pass rate',
                'issue': 'Most drafts are failing QA checks',
                'suggestion': 'Improve intake questions to capture required information',
                'evidence': [],
                'action': {
                    'type': 'update_intake',
                    'field': 'intake_schema',
                    'recommendation': 'Add clearer intake questions for commonly missing fields'
                }
            })
    
    # Sort suggestions by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 2))
    
    return {
        'generated_at': now.isoformat(),
        'period_days': days,
        'total_suggestions': len(suggestions),
        'suggestions': suggestions,
        'summary': {
            'high_priority': len([s for s in suggestions if s['priority'] == 'high']),
            'medium_priority': len([s for s in suggestions if s['priority'] == 'medium']),
            'low_priority': len([s for s in suggestions if s['priority'] == 'low']),
        }
    }
