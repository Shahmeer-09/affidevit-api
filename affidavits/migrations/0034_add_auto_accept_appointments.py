from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0033_add_feedback_summary'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='auto_accept_appointments',
            field=models.BooleanField(
                default=False,
                help_text='When True, new slot bookings are automatically accepted without manual review',
            ),
        ),
    ]
