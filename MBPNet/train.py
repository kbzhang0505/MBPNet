
import time
from options.base_options import BaseOptions
from data import create_dataset
from util.visualizer import Visualizer
from envclass import envclass
from models.ours_model import MBPNModel


if __name__ == '__main__':

    opt = BaseOptions().parse()   # Get training options
    dataset = create_dataset(opt)  # Import dataset to generate dataset path collection

    model = MBPNModel(opt)      # Create models
    model.setup(opt)               # Learning rate setting startup model
    visualizer = Visualizer(opt)   # Display and save printed images
    total_iters = 0                # Total iterations

    myenv = envclass(opt)          #Test model index

    for epoch in range(opt.epoch_count, opt.maxepoch + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  #The number of iterations in the epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:  #Show the frequency of intermediate results of training
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            if total_iters % opt.print_freq == 0:#Print current loss
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, model.get_current_losses(), t_comp)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0: #saves the model and test
            print('saving the model at the end of epoch %d and latest, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            savebest = myenv.env(epoch) #test
            if savebest == "bestmode":
                model.save_networks('bestmode')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.maxepoch, time.time() - epoch_start_time))
        model.update_learning_rate()#Update learning rate
    print('End of train epoch %d ' % epoch)