from django.apps import AppConfig


class AffidavitsConfig(AppConfig):
    name = 'affidavits'

    def ready(self):
        import affidavits.signals
