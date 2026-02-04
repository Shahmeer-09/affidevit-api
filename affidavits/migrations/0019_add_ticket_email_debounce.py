# Generated migration for ticket email debouncing feature

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0018_add_ticket_system'),
    ]

    operations = [
        migrations.AddField(
            model_name='ticket',
            name='last_email_sent_at',
            field=models.DateTimeField(blank=True, help_text='Last time an email notification was sent for this ticket', null=True),
        ),
    ]
