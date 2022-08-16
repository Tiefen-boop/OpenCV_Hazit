# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lines=[]
    line=np.array([[1,2,3],[4,5,6],[7,8,9]])
    # lines=[lines,line]
    lines.append(line)
    print(lines)
    line2=np.array([[19,2,3],[4,5,6],[7,8,9]])
    lines.append(line2)
    print(lines)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
