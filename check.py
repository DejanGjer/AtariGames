import sys
for p in sys.path:
    print(p)
import atari_py
print(atari_py.list_games())