# Generated migration for qa_passed field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='request',
            name='qa_passed',
            field=models.BooleanField(default=False, help_text='True if QA check passed without issues'),
        ),
    ]
