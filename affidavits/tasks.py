"""
Celery tasks for async processing.
AI calls can take 10-30 seconds, so we offload them to background workers.
"""

import time
import logging
from celery import shared_task
from django.db import transaction

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_request_async(self, request_id: int):
    """
    Process a request through AI drafting and QA.
    This is the main async task that handles the AI pipeline.
    
    Args:
        request_id: The ID of the Request to process
        
    Returns:
        dict with status and any error message
    """
    from affidavits.models import Request, RequestEvent, AIRun
    from affidavits.services.ai_service import draft_affidavit, qa_check, analyze_input_suitability
    
    try:
        request = Request.objects.select_related('affidavit_type', 'user').get(id=request_id)
        logger.info(f"Processing request {request.request_code}")
        
        # ===== DEBUG LOGGING =====
        affidavit_type = request.affidavit_type
        logger.info("=" * 80)
        logger.info(f"[CELERY TASK] Request: {request.request_code}")
        logger.info(f"[CELERY TASK] Affidavit Type: {affidavit_type.name} (ID: {affidavit_type.id})")
        logger.info(f"[CELERY TASK] template_html length: {len(affidavit_type.template_html) if affidavit_type.template_html else 0}")
        logger.info(f"[CELERY TASK] template_html first 300 chars: {affidavit_type.template_html[:300] if affidavit_type.template_html else 'EMPTY'}")
        logger.info(f"[CELERY TASK] disallowed_phrases: {affidavit_type.disallowed_phrases}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====
        
        # Log processing start
        RequestEvent.objects.create(
            request=request,
            action=RequestEvent.Action.SUBMITTED,
            details={'task_id': self.request.id}
        )

        # Prepare answers with any clarifications
        final_answers = request.answers_json.copy() if request.answers_json else {}
        user_edits = request.user_edits_json or {}
        clarifications = user_edits.get('clarifications', [])
        
        if clarifications:
            # Format clarifications for the AI
            clarification_text = []
            for c in clarifications:
                clarification_text.append(f"Clarification Q: {c.get('question', '')}\nUser Answer: {c.get('response', '')}")
            
            final_answers['PREVIOUS_CLARIFICATIONS'] = "\n---\n".join(clarification_text)
        
        # Step 0: Input Suitability Check (Pre-draft validation)
        start_time = time.time()
        input_check = analyze_input_suitability(
            answers_json=final_answers,
            template_html=affidavit_type.template_html,
            affidavit_type_name=affidavit_type.name
        )
        check_latency = int((time.time() - start_time) * 1000)
        
        # Log AI run for Input Check (if tokens were used)
        if input_check.get('total_tokens', 0) > 0:
            ai_run_check = AIRun.objects.create(
                request=request,
                node_type=AIRun.NodeType.CLARIFICATION,  # Use CLARIFICATION node type for input checks
                status=AIRun.Status.SUCCESS,
                model_name='gpt-4o',
                prompt_version=request.prompt_version_used or 1,
                prompt_tokens=input_check.get('prompt_tokens', 0),
                completion_tokens=input_check.get('completion_tokens', 0),
                total_tokens=input_check.get('total_tokens', 0),
                latency_ms=check_latency,
                input_json={'answers': final_answers, 'check_type': 'suitability'},
                output_json=input_check,
                raw_response='',
                error_message=''
            )
            ai_run_check.calculate_cost()
            ai_run_check.save()

        # If input is not suitable, stop here and ask for clarification
        if not input_check.get('is_suitable', True):
            request.status = Request.Status.NEEDS_CLARIFICATION
            
            question = input_check.get('clarification_question', 'Please provide more details.')
            example = input_check.get('clarification_example', '')
            
            if example:
                request.clarification_question = f"{question}\n\n{example}"
            else:
                request.clarification_question = question
                
            request.qa_flags_json = input_check.get('issues', [])
            request.save()
            
            RequestEvent.objects.create(
                request=request,
                action=RequestEvent.Action.QA_COMPLETED, # Use QA_COMPLETED as it's a validation step
                details={
                    'reason': 'Input suitability check failed',
                    'question': request.clarification_question
                }
            )
            
            logger.info(f"Request {request.request_code} needs clarification: {request.clarification_question}")
            return {'success': False, 'status': 'needs_clarification', 'question': request.clarification_question}

        # If AI calculated age from DOB, inject it into the answers for the draft
        calculated_age = input_check.get('calculated_age')
        if calculated_age is not None:
            final_answers['_calculated_age'] = calculated_age
            logger.info(f"Using calculated age {calculated_age} from DOB for request {request.request_code}")

        # Step 1: Generate draft - NOW WITH ALL PARAMETERS
        start_time = time.time()
        draft_result = draft_affidavit(
            answers_json=final_answers,
            policy_json=affidavit_type.policy_json,
            affidavit_type_name=affidavit_type.name,
            scenario_library=affidavit_type.scenario_library,
            template_html=affidavit_type.template_html,
            disallowed_phrases=affidavit_type.disallowed_phrases
        )
        draft_latency = int((time.time() - start_time) * 1000)
        
        # Log AI run for drafting
        ai_run_draft = AIRun.objects.create(
            request=request,
            node_type=AIRun.NodeType.DRAFT,
            status=AIRun.Status.SUCCESS if draft_result.get('success') else AIRun.Status.FAILED,
            model_name='gpt-4o-mini',
            prompt_version=request.prompt_version_used or 1,
            prompt_tokens=draft_result.get('prompt_tokens', 0),
            completion_tokens=draft_result.get('completion_tokens', 0),
            total_tokens=draft_result.get('total_tokens', 0),
            latency_ms=draft_latency,
            input_json={'answers': final_answers},
            output_json=draft_result,
            raw_response=draft_result.get('raw_response') or '',
            error_message=draft_result.get('error') or ''
        )
        ai_run_draft.calculate_cost()
        ai_run_draft.save()
        
        if not draft_result.get('success'):
            request.status = Request.Status.NEEDS_REVIEW
            request.qa_flags_json = [{'type': 'draft_error', 'description': draft_result.get('error', 'Draft generation failed')}]
            request.save()
            return {'success': False, 'error': draft_result.get('error', 'Draft generation failed')}
        
        # Update request with draft
        request.draft_text = draft_result.get('draft_text', '')
        request.draft_json = draft_result.get('draft_json', {})
        request.scenario_tags = draft_result.get('scenario_tags', [])
        request.new_scenario_flag = draft_result.get('new_scenario', False)
        request.save()
        
        RequestEvent.objects.create(
            request=request,
            action=RequestEvent.Action.DRAFT_GENERATED,
            details={
                'tokens': ai_run_draft.total_tokens,
                'latency_ms': draft_latency
            }
        )
        
        # Step 2: QA Check
        start_time = time.time()
        qa_result = qa_check(
            draft_text=request.draft_text,
            answers_json=final_answers,
            policy_json=request.affidavit_type.policy_json,
            affidavit_type_name=request.affidavit_type.name
        )
        qa_latency = int((time.time() - start_time) * 1000)
        
        # Log AI run for QA
        ai_run_qa = AIRun.objects.create(
            request=request,
            node_type=AIRun.NodeType.QA,
            status=AIRun.Status.SUCCESS if qa_result.get('success') else AIRun.Status.FAILED,
            model_name='gpt-4o',
            prompt_version=request.prompt_version_used or 1,
            prompt_tokens=qa_result.get('prompt_tokens', 0),
            completion_tokens=qa_result.get('completion_tokens', 0),
            total_tokens=qa_result.get('total_tokens', 0),
            latency_ms=qa_latency,
            input_json={'draft_text': request.draft_text[:500]},  # Truncate for storage
            output_json=qa_result,
            raw_response=qa_result.get('raw_response') or '',
            error_message=qa_result.get('error') or ''
        )
        ai_run_qa.calculate_cost()
        ai_run_qa.save()
        
        if not qa_result.get('success', True):  # Default to True since qa_check returns status not success
            request.status = Request.Status.NEEDS_REVIEW
            request.qa_flags_json = [{'type': 'qa_error', 'description': qa_result.get('error', 'QA check failed')}]
            request.save()
            return {'success': False, 'error': qa_result.get('error', 'QA check failed')}
        
        # Update request with QA results
        qa_status = qa_result.get('status', 'needs_review')
        request.qa_passed = (qa_status == 'approved')
        request.qa_flags_json = qa_result.get('issues') or []
        request.clarification_question = qa_result.get('clarification_question') or ''
        
        # Determine next status based on QA result
        if qa_status == 'approved':
            if request.affidavit_type.default_mode == 'instant' or request.affidavit_type.is_instant_mode:
                request.status = Request.Status.APPROVED
                request.final_text = request.draft_text
            else:
                request.status = Request.Status.NEEDS_REVIEW
        elif qa_status == 'needs_clarification':
            request.status = Request.Status.NEEDS_CLARIFICATION
        else:
            request.status = Request.Status.NEEDS_REVIEW
        
        request.save()
        
        RequestEvent.objects.create(
            request=request,
            action=RequestEvent.Action.QA_COMPLETED,
            details={
                'passed': request.qa_passed,
                'issues_count': len(request.qa_flags_json),
                'tokens': ai_run_qa.total_tokens,
                'latency_ms': qa_latency,
                'status': request.status
            }
        )
        
        logger.info(f"Request {request.request_code} processed successfully. Status: {request.status}")
        
        return {
            'success': True,
            'request_code': request.request_code,
            'status': request.status,
            'qa_passed': request.qa_passed
        }
        
    except Request.DoesNotExist:
        logger.error(f"Request {request_id} not found")
        return {'success': False, 'error': 'Request not found'}
        
    except Exception as exc:
        logger.exception(f"Error processing request {request_id}: {exc}")
        # Retry on transient failures
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        
        # Mark as needs_review after max retries
        try:
            request = Request.objects.get(id=request_id)
            request.status = Request.Status.NEEDS_REVIEW
            request.qa_flags_json = [{'type': 'processing_error', 'description': str(exc)}]
            request.save()
        except Request.DoesNotExist:
            pass
            
        return {'success': False, 'error': str(exc)}


@shared_task(bind=True, max_retries=2)
def generate_pdf_async(self, request_id: int):
    """
    Generate PDF for a completed request.
    
    Args:
        request_id: The ID of the Request to generate PDF for
        
    Returns:
        dict with success status and PDF URL
    """
    from affidavits.models import Request, RequestEvent
    from affidavits.services.pdf_service import generate_affidavit_pdf
    
    try:
        request = Request.objects.select_related('affidavit_type', 'user').get(id=request_id)
        logger.info(f"Generating PDF for request {request.request_code}")
        
        result = generate_affidavit_pdf(request)
        
        if result.get('success'):
            request.pdf_url = result.get('pdf_url', '')
            request.save()
            
            RequestEvent.objects.create(
                request=request,
                action=RequestEvent.Action.PDF_GENERATED,
                details={'pdf_url': request.pdf_url}
            )
            
            logger.info(f"PDF generated for {request.request_code}: {request.pdf_url}")
            return {'success': True, 'pdf_url': request.pdf_url}
        else:
            return {'success': False, 'error': result.get('error', 'PDF generation failed')}
            
    except Request.DoesNotExist:
        logger.error(f"Request {request_id} not found for PDF generation")
        return {'success': False, 'error': 'Request not found'}
        
    except Exception as exc:
        logger.exception(f"Error generating PDF for request {request_id}: {exc}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        return {'success': False, 'error': str(exc)}


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def send_notification_async(self, notification_type: str, request_id: int, **kwargs):
    """
    Send email notifications asynchronously.
    
    Args:
        notification_type: Type of notification ('approval', 'clarification', 'completion', 'submission')
        request_id: The ID of the Request
        **kwargs: Additional notification-specific parameters
    """
    from affidavits.models import Request
    from affidavits.services.notification_service import (
        send_approval_notification,
        send_clarification_notification,
        send_completion_notification,
        send_submission_notification
    )
    
    try:
        request = Request.objects.select_related('affidavit_type', 'user').get(id=request_id)
        
        notification_map = {
            'approval': send_approval_notification,
            'clarification': send_clarification_notification,
            'completion': send_completion_notification,
            'submission': send_submission_notification
        }
        
        send_func = notification_map.get(notification_type)
        if send_func:
            # Completion notification may need stamp object
            if notification_type == 'completion':
                stamp = getattr(request, 'stamp', None)
                result = send_func(request, stamp)
            else:
                result = send_func(request)
            
            if result.get('success'):
                logger.info(f"Sent {notification_type} notification for {request.request_code}")
                return {'success': True}
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"Failed to send {notification_type} notification: {error}")
                # Retry on failure
                if self.request.retries < self.max_retries:
                    raise self.retry(exc=Exception(error))
                return {'success': False, 'error': error}
        else:
            logger.warning(f"Unknown notification type: {notification_type}")
            return {'success': False, 'error': f'Unknown notification type: {notification_type}'}
            
    except Request.DoesNotExist:
        logger.error(f"Request {request_id} not found for notification")
        return {'success': False, 'error': 'Request not found'}
        
    except Exception as exc:
        logger.exception(f"Error sending notification for request {request_id}: {exc}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        return {'success': False, 'error': str(exc)}


@shared_task
def calculate_confidence_metrics():
    """
    Periodic task to recalculate confidence metrics for all affidavit types.
    Run daily via Celery Beat.
    """
    from affidavits.services.dashboard_service import update_all_confidence_metrics
    
    try:
        result = update_all_confidence_metrics()
        logger.info(f"Confidence metrics updated: {result}")
        return result
    except Exception as exc:
        logger.exception(f"Error updating confidence metrics: {exc}")
        return {'success': False, 'error': str(exc)}


@shared_task
def cleanup_expired_locks():
    """
    Periodic task to release expired commissioner locks.
    Run every 5 minutes via Celery Beat.
    """
    from django.utils import timezone
    from affidavits.models import Request
    
    try:
        expired = Request.objects.filter(
            locked_by__isnull=False,
            locked_at__lt=timezone.now() - timezone.timedelta(minutes=30)
        )
        count = expired.count()
        expired.update(locked_by=None, locked_at=None)
        
        logger.info(f"Released {count} expired locks")
        return {'success': True, 'released': count}
        
    except Exception as exc:
        logger.exception(f"Error cleaning up expired locks: {exc}")
        return {'success': False, 'error': str(exc)}


@shared_task
def generate_weekly_learning_report():
    """
    Periodic task to generate weekly learning loop report.
    Run weekly via Celery Beat.
    """
    from affidavits.services.dashboard_service import get_weekly_learning_report
    
    try:
        report = get_weekly_learning_report()
        logger.info(f"Weekly learning report generated: {len(report.get('types', []))} types analyzed")
        return report
    except Exception as exc:
        logger.exception(f"Error generating weekly learning report: {exc}")
        return {'success': False, 'error': str(exc)}
