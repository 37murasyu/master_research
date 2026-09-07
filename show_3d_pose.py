import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
from config import pose_keypoints as _pose_keypoints
plt.style.use('seaborn')


# 値を写さず config から引く。ここでは点数（12）としてしか使わないが、
# 下の骨格エッジは 3D 点列がランドマーク ID の昇順に並んでいることを前提にしている
# （[0][1]=両肩 [6][7]=両腰 …）。新しく保存した CSV はこの並びになる。
pose_keypoints = np.array(_pose_keypoints)

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum, kpts3d in enumerate(p3ds):
        if framenum%2 == 0: continue #skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        #uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])


        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel('y')
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel('z')
        plt.pause(0.1)
        ax.cla()


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
