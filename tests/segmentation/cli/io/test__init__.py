import pytest
from unittest.mock import MagicMock
from specifix.segmentation.cli.io import IOManager


class TestIOManager:

    def test_initialization(self):
        mock_io_manager = MagicMock(spec=IOManager)
        mock_io_manager.input_directory = ''
        mock_io_manager.output_directory = ''

        assert mock_io_manager.input_directory == ''
        assert mock_io_manager.output_directory == ''
