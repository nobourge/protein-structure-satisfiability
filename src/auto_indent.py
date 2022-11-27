import sys
import inspect


class AutoIndent(object):
    def __init__(self, stream):
        self.stream = stream
        self.offset = 0
        self.frame_cache = {}

    def flush(self): pass

    def indent_level(self):
        i = 0
        base = sys._getframe(2)
        f = base.f_back
        while f:
            if id(f) in self.frame_cache:
                i += 1
            f = f.f_back
        if i == 0:
            # clear out the frame cache
            self.frame_cache = {id(base): True}
        else:
            self.frame_cache[id(base)] = True
        return i

    def write(self, stuff):
        indentation = '  ' * self.indent_level()

        def indent(l):
            if l:
                return indentation + l
            else:
                return l

        stuff = '\n'.join([indent(line) for line in stuff.split('\n')])
        self.stream.write(stuff)

# class AutoIndent(object):
#     """Indent debug output based on function call depth."""
#
#     def __init__(self, stream, depth=len(inspect.stack())):
#         """
#         stream is something like sys.stdout.
#         depth is to compensate for stack depth.
#         The default is to get current stack depth when class created.
#
#         """
#         self.stream = stream
#         self.depth = depth
#
#     def indent_level(self):
#         return len(inspect.stack()) - self.depth
#
#     def write(self, data):
#         indentation = '  ' * self.indent_level()
#
#         def indent(i):
#             if i:
#                 return indentation + i
#             else:
#                 return i
#
#         data = '\n'.join([indent(line) for line in data.split('\n')])
#         self.stream.write(data)
