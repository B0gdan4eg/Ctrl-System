"""
Темная тема для приложения
"""


def get_dark_theme() -> str:
    """
    Возвращает CSS для темной темы
    
    Returns:
        CSS стили в виде строки
    """
    return """
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QGroupBox {
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #999;
        }
        QTextEdit {
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
            min-height: 25px;
        }
        QProgressBar {
            border: 2px solid #555;
            border-radius: 5px;
            text-align: center;
            background-color: #2b2b2b;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            background-color: #2b2b2b;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: white;
            border: 1px solid #555;
            padding: 8px 20px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #0d47a1;
        }
        QTabBar::tab:hover {
            background-color: #1565c0;
        }
    """