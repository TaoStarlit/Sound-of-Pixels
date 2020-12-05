import os
import random
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)] # python list for groups.  eg. 2 groups. [None None]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index] # list_sample: all vid

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]. # from the list_sample pick up N sample randomly

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames) # here are from the option.    8 3 25.  8*8, 25, not the begin not the end
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN # for each vid their.  count frame,  path of file

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN # only one integer, one frame from the center

            # absolute frame/audio paths
            for i in range(self.num_frames): # one integer num_frame, here sample 3 frames but stride.  eg.center_frameN=48   0 25 50 --> 48 73 98
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n]) # eg. for each n(vid), there are there frames 48,73,98
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps # 3 frames and one audio?  -0.5 / fps we got the time
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # taozheng check why there is exception
            print("more info"
            print("n and infoN, center_timeN:",n,infoN,center_timeN)
            for n in range(N):
                if audios[n] is not None:
                    print(audios[n].shape)
                else:
                    print(n," audio is not None")
                
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags} # the training data.  mixed_magnitude,  N groups of frames,  N groups of mags
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
