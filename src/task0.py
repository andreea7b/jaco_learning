from teleop_inference import TeleopInference, CONFIG_FILE_DICT

if __name__ == '__main__':
    print "Starting familiarization task."

    TeleopInference('config/task0_methoda_inference_config.yaml')


    #print 'Now you will collect demonstrations to teach this behavior.'
    #TeleopInference(CONFIG_FILE_DICT[1]['d'])
