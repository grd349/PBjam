import numpy as np
import pandas as pd
from pprint import PrettyPrinter


class pretty_printer(PrettyPrinter):
    _dispatch = {}

    def _format_ndarray(self, object, stream, indent, allowance, context, level):
        write = stream.write
        max_width = self._width - indent - allowance
        with np.printoptions(linewidth=max_width):
            string = repr(object)

        lines = string.split('\n')
        string = ('\n' + indent * ' ').join(lines)
        write(string)

    def _pprint_ndarray(self, object, stream, indent, allowance, context, level):
        self._format_ndarray(object, stream, indent, allowance, context, level)

    _dispatch[np.ndarray.__repr__] = _pprint_ndarray

    def _format_dataframe(self, object, stream, indent, allowance, context, level):
        write = stream.write
        max_width = self._width - indent - allowance
        with pd.option_context('display.width', max_width, 'display.max_columns', None):
            string = repr(object)

        lines = string.split('\n')
        string = f'\n{indent*" "}'.join(lines)
        write(string)

    def _pprint_dataframe(self, object, stream, indent, allowance, context, level):
        self._format_dataframe(object, stream, indent, allowance, context, level)

    _dispatch[pd.DataFrame.__repr__] = _pprint_dataframe
