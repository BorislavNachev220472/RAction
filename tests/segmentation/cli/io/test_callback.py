import pytest
from unittest.mock import MagicMock

from specifix.segmentation.cli.io import reset_input_callback

from specifix.segmentation.cli.io import reset_output_callback


class TestResetInputCallback:
    def test_reset_input_callback(self):
        mock_instance = MagicMock()
        mock_instance.inner_directory = 'initial_directory'

        @reset_input_callback
        def dummy_function(instance):
            instance.inner_directory = 'changed_directory'

        dummy_function(mock_instance)

        assert mock_instance.inner_directory == ''

    def test_reset_input_callback_no_change(self):
        mock_instance = MagicMock()
        mock_instance.inner_directory = 'initial_directory'

        @reset_input_callback
        def dummy_function(instance):
            pass

        dummy_function(mock_instance)

        assert mock_instance.inner_directory == ''


class TestResetOutputCallback:
    def test_reset_output_callback(self):
        mock_instance = MagicMock()
        mock_instance.output_folder = 'initial_output_folder'

        @reset_output_callback
        def dummy_function(instance):
            instance.output_folder = 'changed_output_folder'

        dummy_function(mock_instance)

        assert mock_instance.output_folder == ''

    def test_reset_output_callback_no_change(self):
        mock_instance = MagicMock()
        mock_instance.output_folder = 'initial_output_folder'

        @reset_output_callback
        def dummy_function(instance):
            pass

        dummy_function(mock_instance)

        assert mock_instance.output_folder == ''
