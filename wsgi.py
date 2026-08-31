if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .app import app
else:
    from app import app

__all__ = ["app"]
