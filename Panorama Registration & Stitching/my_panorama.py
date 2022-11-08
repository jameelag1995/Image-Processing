import os
import sol4
import time

def main():
    my_vid = 'roomvid.mp4'

    exp_no_ext = my_vid.split('.')[0]
    os.system('mkdir /tmp/dump')
    os.system('mkdir /tmp/dump/%s' % exp_no_ext)
    os.system('ffmpeg -i videos/%s /tmp/dump/%s/%s%%03d.jpg' % (my_vid, exp_no_ext, exp_no_ext))

    s = time.time()
    panorama_generator = sol4.PanoramicVideoGenerator('/tmp/dump/%s/' % exp_no_ext, exp_no_ext, 2100)
    panorama_generator.align_images(translation_only='boat' in my_vid)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

    panorama_generator.save_panoramas_to_video()




if __name__ == '__main__':
    main()