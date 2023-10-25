import gdb

def register_printers(obj):
  from .prettyprinters import pretty_printer
  gdb.printing.register_pretty_printer(obj, pretty_printer)

register_printers(None)
