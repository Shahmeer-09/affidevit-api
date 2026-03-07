from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0034_add_auto_accept_appointments'),
    ]

    operations = [
        migrations.AddField(
            model_name='reviewerfeedback',
            name='answer_fingerprint',
            field=models.JSONField(
                default=dict,
                blank=True,
                help_text='Selector-field answers snapshot for contextual matching',
            ),
        ),
        migrations.AddField(
            model_name='reviewerfeedback',
            name='scenario_tags',
            field=models.JSONField(
                default=list,
                blank=True,
                help_text='Scenario tags detected from the request answers at feedback creation time',
            ),
        ),
    ]
