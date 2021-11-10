from pipeline import reso,stack,pupil,treadmill
from stimulus import stimulus
from stimline import tune
from tqdm import tqdm

"""
schema classes and methods
"""
import numpy as np
import datajoint as dj

schema = dj.schema('microns_L23_nda', create_tables=True)
schema.spawn_missing_classes()

import coregister.solve as cs
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels
from scipy.interpolate import interp1d
import imageio
import cv2
from .func import get_timing_offset,hamming_filter,resize_movie,slice_array


## all tables key sources are fill function 
## look up in collections schema
params = dict(skip_duplicates=True,ignore_extra_fields=True)

##TODO: ADD MESO
##TODO: include in scan table label of type of scan
## animal_id = 21617

@schema
class Scan(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Information on completed scan
    session              : smallint                     # Session ID
    scan_idx             : smallint                     # Scan ID
    ---
    nframes              : int                          # number of frames per scan
    nfields              : tinyint                      # number of fields per scan
    fps                  : float                        # frames per second (Hz)
    scan_type            : varchar(50)                  # scan type/scan protocol
    """
    
    
        
    @property
    def key_source(self):
        return (reso.ScanInfo & self.scan_keys).proj('nframes','nfields','fps')
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Field(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Individual fields of scans
    -> Scan
    field                : smallint                     # Field Number
    ---
    px_width             : smallint                     # field pixels per line
    px_height            : smallint                     # lines per field
    um_width             : float                        # field width (microns)
    um_height            : float                        # field height (microns)
    field_x              : float                        # field x motor coordinates (microns)
    field_y              : float                        # field y motor coordinates (microns)
    field_z              : float                        # field z motor coordinates (microns)
    """
      
    @property
    def key_source(self):
        return ((reso.ScanInfo & Scan.proj() & {'animal_id':8973}) * \
                    reso.ScanInfo.Field).proj('px_width','px_height','um_width','um_height',
                                              field_x='x',field_y='y',field_z='z')
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        
##TODO: can also use automatic tracking        
@schema
class RawManualPupil(dj.Manual):
    """
        Class methods not available outside of BCM pipeline environment     
    """
    definition = """
        # Pupil traces
        -> Scan
        ---
        pupil_min_r          : longblob                     # vector of pupil minor radii  (pixels)
        pupil_maj_r          : longblob                     # vector of pupil major radii  (pixels)
        pupil_x              : longblob                     # vector of pupil x positions  (pixels)
        pupil_y              : longblob                     # vector of pupil y positions  (pixels)
        pupil_times          : longblob                     # vector of times relative to scan start (seconds)
        """
    @property
    def key_source(self):
        return (pupil.FittedPupil & Scan().scan_keys & 'tracking_method = 1')

    @classmethod
    def fill(cls):
        for key in cls.key_source:
            
            pupil_info = (pupil.FittedPupil.Ellipse() & key & 'tracking_method = 1').fetch(order_by='frame_id ASC')
            raw_maj_r,raw_min_r = pupil_info['major_radius'],pupil_info['minor_radius']
            raw_pupil_x = np.array([np.nan if entry is None else entry[0] for entry in pupil_info['center']])
            raw_pupil_y = np.array([np.nan if entry is None else entry[1] for entry in pupil_info['center']])
            pupil_times = (pupil.Eye() & key).fetch1('eye_time')
            offset = (stimulus.BehaviorSync() & key).fetch1('frame_times')[0]
            adjusted_pupil_times = pupil_times - offset


            cls.insert1({'session':key['session'],'scan_idx':key['scan_idx'],
                                'pupil_min_r':raw_min_r,
                                'pupil_maj_r':raw_maj_r,
                                'pupil_x':raw_pupil_x,
                                'pupil_y':raw_pupil_y,
                                'pupil_times':adjusted_pupil_times
                                },**params)
            

    

@schema
class ManualPupil(dj.Manual):
    definition = """
    # Pupil traces
    -> RawManualPupil
    ---
    pupil_min_r          : longblob                     # vector of pupil minor radii synchronized with field 1 frame times (pixels)
    pupil_maj_r          : longblob                     # vector of pupil major radii synchronized with field 1 frame times (pixels)
    pupil_x              : longblob                     # vector of pupil x positions synchronized with field 1 frame times (pixels)
    pupil_y              : longblob                     # vector of pupil y positions synchronized with field 1 frame times (pixels)
    """
    @property
    def key_source(cls):
        return RawManualPupil().proj()
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            stored_pupilinfo = (RawManualPupil() & key).fetch1()
            pupil_times = stored_pupilinfo['pupil_times']
            frame_times,ndepths = (FrameTimes() & key).fetch1('frame_times','ndepths')
            top_frame_scan_times_beh_clock = frame_times[::ndepths]
            raw_pupil_x = [np.nan if entry is None else entry[0] for entry in stored_pupilinfo['pupil_x']]
            raw_pupil_y = [np.nan if entry is None else entry[1] for entry in stored_pupilinfo['pupil_y']]
            pupil_x_interp = interp1d(pupil_times, raw_pupil_x, kind='linear', bounds_error=False, fill_value=np.nan)
            pupil_y_interp = interp1d(pupil_times, raw_pupil_y, kind='linear', bounds_error=False, fill_value=np.nan)
            major_r_interp = interp1d(pupil_times, stored_pupilinfo['pupil_maj_r'], kind='linear', bounds_error=False, fill_value=np.nan)
            minor_r_interp = interp1d(pupil_times,stored_pupilinfo['pupil_min_r'],kind='linear',bounds_error=False,fill_value=np.nan)
            pupil_x = pupil_x_interp(top_frame_scan_times_beh_clock)
            pupil_y = pupil_y_interp(top_frame_scan_times_beh_clock)
            pupil_maj_r = major_r_interp(top_frame_scan_times_beh_clock)
            pupil_min_r = minor_r_interp(top_frame_scan_times_beh_clock)
            cls.insert({**key,'pupil_min_r':pupil_min_r,'pupil_maj_r':pupil_maj_r,'pupil_x':pupil_x,'pupil_y':pupil_y},**params)


@schema
class RawTreadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->Scan
    ---
    treadmill_velocity      : longblob                     # vector of treadmill velocities (cm/s)
    treadmill_timestamps    : longblob                     # vector of times relative to scan start (seconds)
    """
    @property
    def key_source(cls):
        return Scan().scan_keys

    @classmethod
    def fill(cls):
        for key in cls.key_source:

        
            treadmill_info = (treadmill.Treadmill & key).fetch1()
            frame_times_beh = (stimulus.BehaviorSync() & key).fetch1('frame_times')

            adjusted_treadmill_times = treadmill_info['treadmill_time'] - frame_times_beh[0]
            cls.insert1({**key,'treadmill_timestamps':adjusted_treadmill_times,'treadmill_velocity':treadmill_info['treadmill_vel']},**params)

@schema
class Treadmill(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Treadmill traces
    ->RawTreadmill
    ---
    treadmill_speed      : longblob                     # vector of treadmill velocities synchronized with field 1 frame times (cm/s)
    """
    
    @property
    def key_source(self):
        return RawTreadmill().proj()
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:

            tread_time, tread_vel = (RawTreadmill() & key).fetch1('treadmill_timestamps', 'treadmill_velocity')
            tread_time = tread_time.astype(np.float)
            tread_vel = tread_vel.astype(np.float)
            tread_interp = interp1d(tread_time, tread_vel, kind='linear', bounds_error=False, fill_value=np.nan)
            frame_times,depths = (FrameTimes() & key).fetch1('frame_times','ndepths')
            interp_tread_vel = tread_interp(frame_times[::depths])
            treadmill_key = {
                'session': key['session'],
                'scan_idx': key['scan_idx'],
                'treadmill_speed': interp_tread_vel
            }
            cls.insert1(treadmill_key,**params)
        

@schema
class FrameTimes(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # scan times per frame (in seconds, relative to the start of the scan)
    ->Scan
    ---
    frame_times        : longblob            # stimulus frame times for field 1 of each scan, (len = nframes)
    ndepths             : smallint           # number of imaging depths recorded for each scan
    """

    @property
    def key_source(cls):
        return Scan().scan_keys
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            frame_times = (stimulus.BehaviorSync() & key).fetch('frame_times')[0]
            ndepths = len(dj.U('z') &  (reso.ScanInfo().Field() & key))
            cls.insert1({**key,'frame_times':frame_times - frame_times[0],'ndepths':ndepths},**params)

#TODO: only 1 field per depth, 10 depths, bigger diff on frame times
@schema
class Stimulus(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Stimulus presented
    -> Scan
    ---
    movie                : longblob                     # stimulus images synchronized with field 1 frame times (H x W X F matrix)
    """
    @property 
    def key_source(cls):
        return Scan().scan_keys        
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            time_axis = 2
            target_size = (90, 160)
            full_stimulus = None
            full_flips = None

            num_depths = np.unique((reso.ScanInfo.Field & key).fetch('z')).shape[0]
            scan_times = (stimulus.Sync & key).fetch1('frame_times').squeeze()[::num_depths]
            target_hz = 1/np.median(np.diff(scan_times))
            trial_data = ((stimulus.Trial & key) * stimulus.Condition).fetch('KEY', 'stimulus_type', order_by='trial_idx ASC')
            for trial_key,stim_type in zip(tqdm(trial_data[0]), trial_data[1]):
                
                if stim_type == 'stimulus.Clip':
                    djtable = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Clip * stimulus.Movie.Clip * stimulus.Movie)
                    flip_times, compressed_clip, skip_time, cut_after,frame_rate = djtable.fetch1('flip_times', 'clip', 'skip_time', 'cut_after','frame_rate')
                    # convert to grayscale and stack to movie in width x height x time
                    temp_vid = imageio.get_reader(io.BytesIO(compressed_clip.tobytes()), 'ffmpeg')
                    # NOTE: Used to use temp_vid.get_length() but new versions of ffmpeg return inf with this function
                    temp_vid_length = temp_vid.count_frames()
                    movie = np.stack([temp_vid.get_data(i).mean(axis=-1) for i in range(temp_vid_length)], axis=2)
                    assumed_clip_fps = frame_rate
                    start_idx = int(np.float(skip_time) * assumed_clip_fps)
                    print(trial_key)
                    end_idx = int(start_idx + (np.float(cut_after) * assumed_clip_fps))
                
                    movie = movie[:,:,start_idx:end_idx]
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                elif stim_type == 'stimulus.Monet':
                    flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet).fetch1('flip_times', 'movie')
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                elif stim_type == 'stimulus.Trippy':
                    flip_times, movie = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('flip_times', 'movie')
                    movie = resize_movie(movie, target_size, time_axis)
                    movie = hamming_filter(movie, time_axis, flip_times, scan_times)
                    full_stimulus = np.concatenate((full_stimulus, movie), axis=time_axis) if full_stimulus is not None else movie
                    full_flips = np.concatenate((full_flips, flip_times.squeeze())) if full_flips is not None else flip_times.squeeze()
                
                else:
                    raise Exception(f'Error: stimulus type {stim_type} not understood')

            h,w,t = full_stimulus.shape
            interpolated_movie = np.zeros((h, w, scan_times.shape[0]))
            for t_time,i in zip(tqdm(scan_times), range(len(scan_times))):
                idx = (full_flips < t_time).sum() - 1
                if (idx < 0) or (idx >= full_stimulus.shape[2]-2):
                    interpolated_movie[:,:,i] = np.zeros(full_stimulus.shape[0:2])
                else:
                    myinterp = interp1d(full_flips[idx:idx+2], full_stimulus[:,:,idx:idx+2], axis=2)
                    interpolated_movie[:,:,i] = myinterp(t_time)
            

            overflow = np.where(interpolated_movie > 255)
            underflow = np.where(interpolated_movie < 0)
            interpolated_movie[overflow[0],overflow[1],overflow[2]] = 255
            interpolated_movie[underflow[0],underflow[1],underflow[2]] = 0


        

            cls.insert1({'movie':interpolated_movie.astype(np.uint8),'session':key['session'],'scan_idx':key['scan_idx']},**params)

@schema
class Trial(dj.Manual):
    definition = """
    # Information for each Trial
    ->Stimulus
    trial_idx            : smallint                     # index of trial within stimulus
    ---
    type                 : varchar(16)                  # type of stimulus trial
    start_idx            : int unsigned                 # index of field 1 scan frame at start of trial
    end_idx              : int unsigned                 # index of field 1 scan frame at end of trial
    start_frame_time     : double                       # start time of stimulus frame relative to scan start (seconds)
    end_frame_time       : double                       # end time of stimulus frame relative to scan start (seconds)
    frame_times          : longblob                     # full vector of stimulus frame times relative to scan start (seconds)
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    """
    @property 
    def key_source(cls):
        return Scan().scan_keys
    
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            data = ((stimulus.Trial() & key) * stimulus.Condition()).fetch(as_dict=True)
            ndepths = len(dj.U('z') & (reso.ScanInfo().Field() & key))

            offset = get_timing_offset(key)
            frame_times = (stimulus.Sync() & key).fetch1('frame_times')
            field1_pixel0 = frame_times[0::ndepths]


            for idx,trial in enumerate(tqdm(data)):
                trial['start_index'] = None
                trial_flip_times = trial['flip_times'].squeeze()
                
                start_index = (field1_pixel0 < trial_flip_times[0]).sum() - 1
                end_index = (field1_pixel0 < trial_flip_times[-1]).sum() - 1
                start_time = trial_flip_times[0]
                end_time = trial_flip_times[-1]
                if(idx > 0):
                    if(data[idx-1]['end_idx'] == start_index):
                        print('ding')
                        t0 = data[idx-1]
                        med = (start_time+t0['end_time'])/2
                        nearest_frame_start = np.argmin(np.abs(field1_pixel0 - med))
                        data[idx-1]['end_idx'] = nearest_frame_start-1
                        trial['start_idx'] = nearest_frame_start
                

                adj_start_time = start_time - offset
                adj_end_time = end_time - offset
                adj_trial_flip_times = trial_flip_times - offset
                data[idx]['start_frame_time'] = adj_start_time 
                data[idx]['end_frame_time'] = adj_end_time 
                data[idx]['start_idx'] = trial['start_index'] if trial['start_index'] is not None else start_index 
                data[idx]['end_idx'] = end_index
                data[idx]['frame_times'] = adj_trial_flip_times
                data[idx]['type'] = data[idx]['stimulus_type']
                        
            cls.insert(data,**params)


@schema
class Monet(dj.Manual):
    definition = """
    # pink noise with periods of motion and orientation
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    fps                  : decimal(5,2)                 # display refresh rate
    moving_noise_version : smallint                     # algorithm version; increment when code changes
    rng_seed             : double                       # random number generate seed
    tex_ydim             : smallint                     # texture dimension (pixels) 
    tex_xdim             : smallint                     # texture dimension (pixels) 
    spatial_freq_half    : float                        # spatial frequency modulated to 50 percent (cy/deg) 
    spatial_freq_stop    : float                        # spatial lowpass cutoff (cy/deg)
    temp_bandwidth       : float                        # temporal bandwidth of the stimulus (Hz)
    ori_on_secs          : float                        # duration of moving/oriented stimulus (seconds)
    ori_off_secs         : float                        # duration of unmoving/unoriented stimulus (seconds)
    n_dirs               : smallint                     # number of directions
    ori_bands            : tinyint                      # orientation width expressed in units of 2*pi/n_dirs
    ori_modulation       : float                        # mixin-coefficient of orientation biased noise
    speed                : float                        # (degrees/s)
    x_degrees            : float                        # degrees across x if screen were wrapped at shortest distance
    y_degrees            : float                        # degrees across y if screen were wrapped at shortest distance
    directions           : blob                         # directions in periods of motion (degrees)
    onsets               : blob                         # moving period onset times (seconds) 
    movie                : longblob                     # rendered uint8 movie (H X W X T)
    """
    
    @property
    def key_source(self):
        return stimulus.Monet & (stimulus.Trial & {'animal_id':8973} & Scan)

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

##TODO: add Trippy
##TODO: add Clip


@schema
class MeanIntensity(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # mean intensity of imaging field over time
    ->Field
    ---
    intensities    : longblob                     # mean intensity
    """
    
    @property
    def key_source(self):
        return reso.Quality.MeanIntensity & {'animal_id':8973,'channel':1} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)
        
        
        

@schema
class SummaryImages(dj.Manual):
    definition = """
    ->Field
    channel        : tinyint                      # green (1) or red (2) channel
    ---
    correlation    : longblob                     # average image
    average        : longblob                     # correlation image
    """
    
    @property
    def key_source(self):
        return reso.SummaryImages.Correlation.proj(correlation='correlation_image') * \
               reso.SummaryImages.Average.proj(average='average_image') & {'animal_id': 8973} & Scan
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source,**params)

@schema
class Stack(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # all slices of each stack after corrections.
    stack_session        : smallint                     # session index for the mouse
    stack_idx            : smallint                     # id of the stack
    ---
    motor_z              : float                        # (um) center of volume in the motor coordinate system (cortex is at 0)
    motor_y              : float                        # (um) center of volume in the motor coordinate system
    motor_x              : float                        # (um) center of volume in the motor coordinate system
    px_depth             : smallint                     # number of slices
    px_height            : smallint                     # lines per frame
    px_width             : smallint                     # pixels per line
    um_depth             : float                        # depth in microns
    um_height            : float                        # height in microns
    um_width             : float                        # width in microns
    surf_z               : float                        # (um) depth of first slice - half a z step (cortex is at z=0)
    """
    
    
    
    @property
    def key_source(self):
        fetch_str = stack.CorrectedStack.heading.secondary_attributes[3:]
        return stack.CorrectedStack.proj(*fetch_str,motor_x='x',motor_y='y',motor_z='z',
                                          stack_session='session') & self.stack_key
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Registration(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # align a 2-d scan field to a stack with affine matrix learned via gradient ascent
    ->Stack
    ->Field
    ---
    a11                  : float                        # (um) element in row 1, column 1 of the affine matrix
    a21                  : float                        # (um) element in row 2, column 1 of the affine matrix
    a31                  : float                        # (um) element in row 3, column 1 of the affine matrix
    a12                  : float                        # (um) element in row 1, column 2 of the affine matrix
    a22                  : float                        # (um) element in row 2, column 2 of the affine matrix
    a32                  : float                        # (um) element in row 3, column 2 of the affine matrix
    reg_x                : float                        # z translation (microns)
    reg_y                : float                        # y translation (microns)
    reg_z                : float                        # z translation (microns)
    score                : float                        # cross-correlation score (-1 to 1)
    reg_field            : longblob                     # extracted field from the stack in the specified position
    """
    
    @property
    def key_source(self):
        fetch_str = stack.Registration.Affine.heading.secondary_attributes
        return stack.Registration.Affine.proj(*fetch_str,session='scan_session') & {'animal_id':8973} & Stack & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


##TODO: Check if exists
@schema
class Coregistration(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # transformation solutions between 2P stack and EM stack and vice versa from the Allen Institute
    ->Stack
    transform_id            : int                          # id of the transform
    ---
    version                 : varchar(256)                 # coordinate framework
    direction               : varchar(16)                  # direction of the transform (EMTP: EM -> 2P, TPEM: 2P -> EM)
    transform_type          : varchar(16)                  # linear (more rigid) or spline (more nonrigid)
    transform_args=null     : longblob                     # parameters of the transform
    transform_solution=null : longblob                     # transform solution
    """
    
    @property
    def key_source(self):
        return m65p3.Coregistration()
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)



@schema
class Segmentation(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Different mask segmentations
    ->Field
    mask_id         :  smallint
    ---
    pixels          : longblob      # indices into the image in column major (Fortran) order
    weights         : longblob      # weights of the mask at the indices above
    """
    
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6}

    @property
    def key_source(self):
        return reso.Segmentation.Mask & self.segmentation_key & Field

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Fluorescence(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # fluorescence traces before spike extraction or filtering
    -> Segmentation
    ---
    trace                   : longblob #fluorescence trace 
    """
    
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6}

    @property
    def key_source(self):
        return reso.Fluorescence.Trace & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source,**params)


@schema
class ScanUnit(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # single unit in the scan
    -> Scan
    unit_id                 : int               # unique per scan
    ---
    -> Fluorescence
    um_x                : smallint      # centroid x motor coordinates (microns)
    um_y                : smallint      # centroid y motor coordinates (microns)
    um_z                : smallint      # centroid z motor coordinates (microns)
    px_x                : smallint      # centroid x pixel coordinate in field (pixels
    px_y                : smallint      # centroid y pixel coordinate in field (pixels
    ms_delay            : smallint      # delay from start of frame (field 1 pixel 1) to recording ot his unit (milliseconds)
    """
    
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6}

    
    @property
    def key_source(self):
        return (reso.ScanSet.Unit * reso.ScanSet.UnitInfo) & self.segmentation_key & Field  
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class Activity(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # activity inferred from fluorescence traces
    -> ScanUnit
    ---
    trace                   : longblob  #spike trace
    """
    
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6, 'spike_method': 5}

    @property
    def key_source(self):
        return reso.Activity.Trace & self.segmentation_key & Field

    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)

@schema
class StackUnit(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # centroids of each unit in stack coordinate system using affine registration
    -> Registration
    -> ScanUnit
    ---
    motor_x            : float    # centroid x stack coordinates with motor offset (microns)
    motor_y            : float    # centroid y stack coordinates with motor offset (microns)
    motor_z            : float    # centroid z stack coordinates with motor offset (microns)
    stack_x            : float    # centroid x stack coordinates (microns)
    stack_y            : float    # centroid y stack coordinates (microns)
    stack_z            : float    # centroid z stack coordinates (microns)
    """
    
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6}
    
    @property
    def key_source(self):
        return reso.StackCoordinates.UnitInfo & self.segmentation_key & Stack.proj() & Field

    
    @classmethod
    def fill(cls):
        stack_unit = (cls.key_source*Stack).proj(stack_x = 'round(stack_x - motor_x + um_width/2, 2)', 
                                                 stack_y = 'round(stack_y - motor_y + um_height/2, 2)', 
                                                 stack_z = 'round(stack_z - motor_z + um_depth/2, 2)')
        cls.insert((cls.key_source.proj(motor_x='stack_x', 
                                        motor_y='stack_y', 
                                        motor_z='stack_z') * stack_unit), **params)


#TODO: verify retinotopy scan, widefield retinotopy and ROI
@schema 
class AreaMembership(dj.Manual):
    definition = """
    -> ScanUnit
    ---
    brain_area          : char(10)    # Visual area membership of unit
    
    """
    @property
    def key_source(cls):
        return ScanUnit()
    @classmethod
    def fill(cls):
        for key in cls.key_source:
            cls.insert1({**key,'brain_area':'V1'},**params)

@schema
class MaskClassification(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # classification of segmented masks using CaImAn package
    ->Segmentation
    ---
    mask_type                 : varchar(16)                  # classification of mask as soma or artifact
    """

    @property
    def key_source(self):
        return reso.MaskClassification.Type.proj(mask_type='type') & {'animal_id': 8973, 'segmentation_method': 6} & Scan

    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, **params)


@schema
class Oracle(dj.Manual):
    """
    Class methods not available outside of BCM pipeline environment
    """
    definition = """
    # Leave-one-out correlation for repeated videos in stimulus.
    -> ScanUnit
    ---
    trials               : int                          # number of trials used
    pearson              : float                        # per unit oracle pearson correlation over all movies
    """
    segmentation_key = {'animal_id': 8973, 'segmentation_method': 6, 'spike_method': 5}
    
    @property
    def key_source(self):
        return tune.MovieOracle.Total & self.segmentation_key & Field
    
    @classmethod
    def fill(cls):
        cls.insert(cls.key_source, )

