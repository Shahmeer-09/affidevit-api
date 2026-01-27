# Generated migration for pdf_preferences field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0002_add_qa_passed_field'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='pdf_preferences',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Commissioner-specific PDF formatting preferences'
            ),
        ),
    ]
