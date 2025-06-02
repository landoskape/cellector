import pytest
from PyQt5.QtCore import qInstallMessageHandler, QtMsgType


def silent_qt_warnings(msg_type, context, message):
    if msg_type == QtMsgType.QtWarningMsg:
        if any(
            substr in message
            for substr in [
                "QFontDatabase",
                "QOpenGLWidget",
                "createPlatformOpenGLContext",
                "propagateSizeHints",
            ]
        ):
            return  # suppress these warnings
    print(message)  # show all others


@pytest.fixture(scope="session", autouse=True)
def suppress_qt_warnings():
    qInstallMessageHandler(silent_qt_warnings)
