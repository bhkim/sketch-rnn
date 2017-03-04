import numpy as np
import tensorflow as tf

import argparse
import time
import os
import sys
import cPickle
import shlex, subprocess

from utils import SketchLoader
from model_skipconn import Model
#from model import Model
#from sample_func import sample

import threading
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=256, 
                           help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
    parser.add_argument('--skip_conn', type=bool, default=True,
                     help='adding vertical skip connections: input-to-hiddens, hiddens-to-output ')
    parser.add_argument('--resid_conn', type=bool, default=True,
                     help='adding residual connections between recurrent layers')
    parser.add_argument('--model', type=str, default='gru',
                     help='rnn, gru, lstm, or hyperlstm')
    parser.add_argument('--batch_size', type=int, default=64,  #100,
                     help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=500, #300,
                     help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50, #500
                     help='number of epochs')
    parser.add_argument('--save_every', type=int, default=250, # 250
                     help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.0,  #10.0, 
                     help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                     help='decay rate for rmsprop')
    parser.add_argument('--num_mixture', type=int, default=24, #24
                     help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=1.0,  #15.0,  # BHKIM: len of the max-axis will be normalized as 1.0
                     help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.9,  #0.8
                     help='dropout keep probability')
    parser.add_argument('--stroke_importance_factor', type=float, default=200, # 200
                     help='relative importance of pen status over mdn coordinate accuracy')
    parser.add_argument('--dataset_name', type=str, default="hangul",  # hangul  #kanji
                     help='name of directory containing training data')
    args = parser.parse_args()
    train(args)

#########################################################################
# aux functions

def email_report(mail_subject, fileToSend=None):

    sender_email = 'abc@abc.net' 
    receiver_email = 'abc@abc.net'
    smtp_server = 'smtp.abc.net'
    smtp_port = 465
    sender_id = myid
    sender_pw = mypw
    
    # send reports via e-mail
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = mail_subject
    msg.preamble = mail_subject
    
    if fileToSend is not None and len(fileToSend) > 0:
        ctype, encoding = mimetypes.guess_type(fileToSend)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"

        maintype, subtype = ctype.split("/", 1)

        if maintype == "text":
            fp = open(fileToSend)
            # Note: we should handle calculating the charset
            attachment = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "image":
            fp = open(fileToSend, "rb")
            attachment = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "audio":
            fp = open(fileToSend, "rb")
            attachment = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(fileToSend, "rb")
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(fp.read())
            fp.close()
            encoders.encode_base64(attachment)
        if len(fileToSend) > 0 :
            attachment.add_header("Content-Disposition", "attachment", filename=fileToSend)
            msg.attach(attachment)
    
    s = smtplib.SMTP_SSL(smtp_server,smtp_port)
    s.login(myid, mypw)
    try:
        s.sendmail(sender_email, receiver_email, msg.as_string())
    except:
        pass
    s.quit()
    return True

def sample_from_models_email_report(modelfolder, targetfile, mail_subject):
    
    #'''
    # run string command
    time.asctime(time.gmtime())
    
    def subprocess_open(command):
        args = shlex.split(command)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (stdoutdata, stderrdata) = popen.communicate()
        return stdoutdata, stderrdata
    
    #'''
    command = 'python sample.py --sample_length 600 --filename {} --scale_factor 60.0 --num_picture 30 --dataset_name {} --temperature 0.001 --picture_size 120 --num_col 6 --stroke_width 3'.format(targetfile, modelfolder)
    stdoutdata, stderrdata = subprocess_open(command)
    print(stdoutdata)
    
    while True:
        if os.path.exists(targetfile+'.svg'):
            break
        else:
            sys.stdout.write('.')
            time.sleep(1)
    '''
    th = threading.Thread(target=sample, args=(targetfile, 600, 120, 60.0, 30, 6, modelfolder, 1, 3.0, 0.001))
    #sample(sample_length=600, filename=targetfile, scale_factor=60.0, num_picture=30, dataset_name=modelfolder,
    #       temperature=0.001, picture_size=120, num_col=6, stroke_width=3.0)
    th.start()
    th.join()
    #'''
        
    return email_report(mail_subject, targetfile+'.svg')    

#########################################################################
# this is the main function
def train(args):
    data_loader = SketchLoader(args.batch_size, args.seq_length, args.data_scale, args.dataset_name)

    dirname = os.path.join('save', args.dataset_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    model = Model(args, infer=False)

    b_processed = 0
    
    # added by bhkim: 2016-11-17
    min_train_loss = 0.0
    #learn_rate = tf.Variable(args.learning_rate)
    def save_model(sess, tot_loss, shape_loss, pen_loss):  # modified by bhkim to get three losses as parameters: 2016-11-17
        checkpoint_path = os.path.join('save', args.dataset_name, 'model.ckpt')
        #saver.save(sess, checkpoint_path, global_step = b_processed)
        savefilename = "%s_%.2f_%.2f_%.4f" % (checkpoint_path, tot_loss, shape_loss, pen_loss)
        saver.save(sess, savefilename, global_step = b_processed)
        print "model saved to {}".format(savefilename)
        return savefilename
    
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run() #initialize_all_variables().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)  # all_variables()

        # load previously trained model if appilcable
        ckpt = tf.train.get_checkpoint_state(os.path.join('save', args.dataset_name))
        if ckpt:
            print "loading last model: ",ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
            startidx = ckpt.model_checkpoint_path.find('-')   # ex) model.ckpt_-5.52_-6.49_0.9670-50813
            min_train_loss = float( ckpt.model_checkpoint_path[ 
                                        startidx:ckpt.model_checkpoint_path.find('_', startidx)] )
            print "starting from the minimum loss: {}".format(min_train_loss)

        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_index_pointer()
            
            ###
            # state = model.initial_state.eval() : this does not works in case of 'tuple' as in LSTM
            # They suggest using sess.run(x) instead (https://github.com/hunkim/word-rnn-tensorflow/issues/9)
            state = sess.run(model.initial_state) 
                
            while data_loader.epoch_finished == False:
                start = time.time()
                input_data, target_data = data_loader.next_batch()
                
                ###
                feed_dict = {model.input_data: input_data, model.target_data: target_data, model.initial_state: state}
                train_loss, shape_loss, pen_loss, state, _ = \
                        sess.run([model.cost, model.cost_shape, model.cost_pen, model.final_state, model.train_op], feed_dict)
                end = time.time()
                b_processed += 1
                print "{}/{} (epoch {} batch {}), cost = {:.2f} ({:.2f}+{:.4f}), time/batch = {:.2f}" \
                .format(data_loader.pointer + e * data_loader.num_samples,
                        args.num_epochs * data_loader.num_samples,
                        e, b_processed ,train_loss, shape_loss, pen_loss, end - start)
                # assert( train_loss != np.NaN or train_loss != np.Inf) # doesn't work.
                assert( train_loss < 30000) # if dodgy loss, exit w/ error.
                #if (b_processed) % args.save_every == 0 and ((b_processed) > 0):
                #    save_model(train_loss, shape_loss, pen_loss)
                    
                # added by bhkim: 2016-11-17    
                if train_loss < min_train_loss:
                    savefilename = save_model(sess, train_loss, shape_loss, pen_loss)
                    min_train_loss = train_loss
                    
                    filesToSend = '{}'.format(savefilename)
                    mail_subject = '%s [%d]*%d layer- ep%d, data: %s, cost = %.2f (%.2f + %.4f)' % (args.model, args.rnn_size, args.num_layers, e, args.dataset_name, train_loss, shape_loss, pen_loss)
                    
                    datasetname = args.dataset_name
                    #th = threading.Thread(target=sample_from_models_email_report, args=(datasetname, filesToSend, mail_subject))
                    #th.start()
                    #th.join()
                    #sample_from_models_email_report(modelfolder=args.dataset_name, targetfile=filesToSend, mail_subject=mail_subject)  
                    email_report(mail_subject, None)
                    
        #save_model(sess, train_loss, shape_loss, pen_loss)

if __name__ == '__main__':
    main()


