from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0017_add_site_settings'),
    ]

    operations = [
        migrations.CreateModel(
            name='Ticket',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subject', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('category', models.CharField(choices=[('technical', 'Technical Issue'), ('billing', 'Billing'), ('legal', 'Legal Question'), ('other', 'Other')], default='other', max_length=20)),
                ('status', models.CharField(choices=[('open', 'Open'), ('in_progress', 'In Progress'), ('resolved', 'Resolved'), ('closed', 'Closed')], default='open', max_length=20)),
                ('priority', models.CharField(choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('urgent', 'Urgent')], default='medium', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('resolved_at', models.DateTimeField(blank=True, null=True)),
                ('request', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='tickets', to='affidavits.request')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tickets', to='affidavits.user')),
            ],
            options={
                'db_table': 'tickets',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='TicketMessage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('message', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('is_internal', models.BooleanField(default=False, help_text='Internal note for admins only')),
                ('sender', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ticket_messages', to='affidavits.user')),
                ('ticket', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='messages', to='affidavits.ticket')),
            ],
            options={
                'db_table': 'ticket_messages',
                'ordering': ['created_at'],
            },
        ),
        migrations.CreateModel(
            name='TicketAttachment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='tickets/%Y/%m/%d/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('ticket', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='attachments', to='affidavits.ticket')),
            ],
            options={
                'db_table': 'ticket_attachments',
            },
        ),
        migrations.AlterField(
            model_name='requestevent',
            name='action',
            field=models.CharField(choices=[('created', 'Created'), ('submitted', 'Submitted'), ('draft_generated', 'Draft Generated'), ('qa_completed', 'QA Completed'), ('viewed', 'Viewed'), ('edited', 'Edited'), ('approved', 'Approved'), ('rejected', 'Rejected'), ('completed', 'Completed (Stamped)'), ('locked', 'Locked'), ('unlocked', 'Unlocked'), ('pdf_generated', 'PDF Generated'), ('pdf_downloaded', 'PDF Downloaded'), ('commissioner_opened', 'Commissioner Opened'), ('commissioner_changed', 'Commissioner Changed'), ('ticket_created', 'Ticket Created'), ('ticket_updated', 'Ticket Updated')], db_index=True, max_length=30),
        ),
    ]
