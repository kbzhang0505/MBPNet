

from options.base_options import BaseOptions
from envclass import envclass


if __name__ == '__main__':

    opt = BaseOptions().parse()

    myenv=envclass(opt)

    myenv.envsave("bestmode")