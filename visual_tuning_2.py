##visual_tuning_2.py
##by Ryan Neely
##a program to acquire neural data during the presentation of 
##drifting grating visual stimuli. 
##Designed to run with TDT hardware and ActiveX controls as 
##well as psychopy in order to generate stimuli.

from psychopy import visual
from psychopy import core
from psychopy import info
import TDT_control
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from multiprocessing.pool import ThreadPool
import pickle

##grating variables
#directions of movement (list). -1 is the baseline gray screen
##***NOTE: for plotting to be correct, the gray screen value should be last in the array!!!***
DIRECTIONS = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, -1])
##contrast values (list). -1 is the baseline gray screen
CONTRASTS = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -1])
#the spatial frequency of the grating in cycles/degree
SPATIAL_FREQ = 0.04
##amount of time to display the gray screen in seconds
GRAY_TIME = 2.0
##amount of time to display the drifting grating in seconds
DRIFT_TIME = 4.0
#the number of times to repeat the full set of gratings
NUM_SETS = 8
##the location of the TDT circuit file to load
CIRCUIT_LOC = r"C:\TDT\OpenEx\MyProjects\visual_tuning_2\RCOCircuits\V1_tuning_circuit.rcx"
##the location to save the data
SAVE_LOC = r"C:\Data\V07\TEST.hdf5"
#r"C:\Users\TDT-RAT\Desktop\test1.hdf5"
##the location to save the plot data
DICT1_LOC = r"C:\Data\V06\session_7.2_tuning.p"
DICT2_LOC = r"C:\Data\V06\session_7.2_norm.p"
##create an HDF5 file in which to save the data
dataFile = h5py.File(SAVE_LOC, 'w-')
##load the spcified circuit file and connect to the processor
"""
NOTE: Loading the wrong file isn't immediately obvious and can cause
a lot of headaches!!
"""
RZ2 = TDT_control.RZ2(CIRCUIT_LOC)
##load the RPvdsEx circuit locally
RZ2.load_circuit(local = True, start = False)
##get the processor sampling rate
fs = RZ2.get_fs()
#print 'sample rate is: ' + str(fs)
##the number of samples to take from the TDT (duration of each stim rep)
num_samples = int(np.ceil(fs*(DRIFT_TIME+2*GRAY_TIME)))

## a helper function to get the names of all the sorted units in a file set
def get_sorted_names(fIn):
	##get the names from one set as a start
	unit_names = TDT_control.parse_sorted(np.asarray(fIn['orientation']['set_1']['0'][0,:,:])).keys()
	##check all instances, and if there is a new addition, add it to the master list
	for setN in range(NUM_SETS):
		for oriN in DIRECTIONS:
			units_present = TDT_control.parse_sorted(np.asarray(fIn['orientation']['set_'+str(setN+1)][str(oriN)][0,:,:])).keys()
			for unit in units_present:
				if unit not in unit_names:
					print "adding a unit - " + unit
					unit_names.append(unit)
	return unit_names

##a function to generate polar plots and save data for direction tuning
def plot_direction_tuning(file_path):
	##re-define directions in the correct order 
	directions_2 = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, -1])
	##open the file
	fIn = h5py.File(file_path, 'r')
	##first, just figure out how many sorted units there are.
	unit_names = get_sorted_names(fIn)
	num_units = len(unit_names)
	##decide what sample range corresponds to the stimuli presentation
	drift_samples = np.arange(np.floor(fs*GRAY_TIME),np.ceil(fs*GRAY_TIME+fs*DRIFT_TIME)).astype(int)
	##make a data dictionary; allocate memory
	tuning_dict = {}
	for name in unit_names:
		tuning_dict[name] = np.zeros((directions_2.size, drift_samples.size))
	##go through each set
	for nSet in range(NUM_SETS):
		##...and each unit
		for name in unit_names:
			##...and each angle
			## add the data from each direction to the dictionary
			for n, angle in enumerate(directions_2):
				data = TDT_control.parse_sorted(np.asarray(fIn['orientation']['set_'+str(nSet+1)][str(angle)][0,:,:]))
				try:
					tuning_dict[name][n,:] += data[name][drift_samples]
				except KeyError:
					##if the parsing function didn't pick up this unit on this trial, it means there were no spikes
					##so just "add" zeros 
					"no spikes for unit " + name 
					pass
	##do some more work to normalize the values
	norm_dict = {}
	baseline_dict = {}
	response_dict = {}
	for name in unit_names:
		current_set = tuning_dict[name]
		##calculate the spike rates
		rates = []
		for d in range(current_set.shape[0]):
			rates.append(float(current_set[d,:].sum())/DRIFT_TIME/NUM_SETS)
		rates = np.asarray(rates)
		baseline_dict[name] = rates[-1]
		response_dict[name] = max(rates)
		##normalize everything to the control rate (assuming here that it's the last in the list)
		if rates[-1] != 0:
			rates = rates/rates[-1]
		else: 
			print "rate for control is zero."
		norm_dict[name] = rates
	##save the data dictionaries with pickle
	#pickle.dump(tuning_dict, open(DICT1_LOC, 'wb'))
	#pickle.dump(tuning_dict, open(DICT2_LOC, 'wb'))
	##plot everything! (finally)
	
	for name in unit_names:
		plt.figure()
		ax = plt.subplot(111, polar = True)
		ax.plot(np.radians(directions_2[0:-1]), norm_dict[name][0:-1], color = np.random.rand(3,), 
			lw = 3, label = 'magnitude of fr above baseline')
		ax.set_rmax(norm_dict[name].max())
		ax.grid(True)
		ax.legend()
		#plt.text(0,1,"baseline= "+str(baseline_dict[name])+"Hz", horizontalalignment = 'left', verticalalignment = 'bottom')
		#plt.text(0,0.5,"max response= "+ str(response_dict[name])+"Hz", horizontalalignment = 'right', verticalalignment = 'bottom')
		ax.set_title(name)
	plt.show()

def run_orientations(plot = False):
	##double check that the TDT processors are connected and the circuit is running
	if RZ2.get_status() != 7:
		raise SystemError, "Check RZ2 status!"
	print "Running orientation presentation."
	##create a file group for orientation data
	ori_group = dataFile.create_group("orientation")
	##create a window for the stimuli
	myWin = visual.Window([1280,1024] ,monitor="rosewill_r910e", units="deg", fullscr = True, screen = 1)
	##get the system/monitor info (interested in the refresh rate)
	#print "Testing monitor refresh rate..."
	#sysinfo = info.RunTimeInfo(author = 'Ryan', version = '1.0', win = myWin, 
	#	refreshTest = 'grating', userProcsDetailed = False, verbose = False)
	##get the length in ms of one frame
	#frame_dur = float(sysinfo['windowRefreshTimeMedian_ms'])
	frame_dur = 11.7 #calculated beforehand

	##create a grating object
	grating = visual.GratingStim(win=myWin, mask = None, size=230,
	                             pos=[0,0], sf=SPATIAL_FREQ, ori = 0, units = 'deg')
	##calculate the number of frames needed to produce the correct display time
	num_frames = int(np.ceil((DRIFT_TIME*1000.0)/frame_dur))
	##set RZ2 recording time parameters
	RZ2.set_tag("samples", num_samples)
	##generate the stimuli
	for setN in range(NUM_SETS):
		print "Beginning set " + str(setN+1) + " of " + str(NUM_SETS)
		##shuffle the orientations
		np.random.shuffle(DIRECTIONS)
		##create a file group for this set
		set_group = ori_group.create_group("set_" + str(setN+1))
		for repN in range(DIRECTIONS.size):
			##make sure you are still connected to the RZ2
			if RZ2.get_status == 0:
				raise SystemError, "Hardware connection lost"
			##create a dataset for this orientation
			dset = set_group.create_dataset(str(DIRECTIONS[repN]), (3,16,num_samples), dtype = 'f')
			##initialize thread pool to stream data
			#dpool = ThreadPool(processes = 3)
			##set the contrast to zero:
			grating.contrast = 0.0
			grating.draw()
			myWin.flip()
			##trigger the RZ2 to begin recording
			RZ2.send_trig(1)
			##start threads
			#sort_thread = dpool.apply_async(RZ2.stream_data, ("sorted", num_samples, 16, "I32", "int"))
			#spk_thread = dpool.apply_async(RZ2.stream_data, ("spkR", num_samples, 16, "F32", "float"))
			#lfp_thread = dpool.apply_async(RZ2.stream_data, ("lfpR", num_samples, 16, "F32", "float"))
			##pause for the Gray time
			core.wait(GRAY_TIME)
			##make sure this isn't a gray trial
			if DIRECTIONS[repN] != -1:
				##adjust the orientation
				grating.ori = DIRECTIONS[repN]
				##bring the contrast back to 100%
				grating.contrast = 1.0
				##draw the stimuli and update the window
				print "Showing orientation " + str(DIRECTIONS[repN])
				for frameN in range(num_frames):
					grating.phase = (0.026*frameN, 0.0)
					grating.draw()
					myWin.flip()
			else:
				##continue to display gray screen
				print "Showing zero contrast control"
				core.wait(DRIFT_TIME)
			##set the contrast to zero:
			grating.contrast = 0.0
			grating.draw()
			myWin.flip()
			##pause for the specified time
			core.wait(GRAY_TIME)
			##now save the data to the hdf5 file
			#sorted spikes
			dset[0,:,:] = RZ2.stream_data("sorted", num_samples, 16, "I32", "int")
			#dset[0,:,:] = sort_thread.get()
			#raw spikes
			dset[1,:,:] = RZ2.stream_data("spkR", num_samples, 16, "F32", "float")
			#dset[1,:,:] = spk_thread.get()
			#raw lfp
			dset[2,:,:] = RZ2.stream_data("lfpR", num_samples, 16, "F32", "float")
			#dset[2,:,:] = lfp_thread.get()
			##clean up
			#dpool.close()
			#dpool.join()

	print "Orientation test complete."
	myWin.close()
	dataFile.close()
	RZ2.stop()
	if plot:
		plot_direction_tuning(SAVE_LOC)		


"""
Basically a repeat of the above function, but using the contrast
values instead. The orientation parameter defines which orientation to run the 
contrasts for. 
"""
def run_contrasts(orientation):
	##double check that the TDT processors are connected and the circuit is running
	if RZ2.get_status() != 7:
		raise SystemError, "Check RZ2 status!"
	print "Running contrast presentation."
	##create a file group for orientation data
	contrast_group = dataFile.create_group("contrast@"+str(orientation))
	##create a window for the stimuli
	myWin = visual.Window([1280,1024] ,monitor="rosewill_r910e", units="deg", fullscr = True, screen = 1)
	##get the system/monitor info (interested in the refresh rate)
	#print "Testing monitor refresh rate..."
	#sysinfo = info.RunTimeInfo(author = 'Ryan', version = '1.0', win = myWin, 
	#	refreshTest = 'grating', userProcsDetailed = False, verbose = False)
	##get the length in ms of one frame
	#frame_dur = float(sysinfo['windowRefreshTimeMedian_ms'])
	frame_dur = 13.33 #calculated beforehand

	##create a grating object
	grating = visual.GratingStim(win=myWin, mask = 'circle', size=80,
	                             pos=[0,0], sf=SPATIAL_FREQ, ori = orientation, units = 'deg')
	##calculate the number of frames needed to produce the correct display time
	num_frames = int(np.ceil((DRIFT_TIME*1000.0)/frame_dur))
	##set RZ2 recording time parameters
	RZ2.set_tag("samples", num_samples)
	##generate the stimuli
	for setN in range(NUM_SETS):
		##shuffle the orientations
		np.random.shuffle(CONTRASTS)
		##create a file group for this set
		set_group = contrast_group.create_group("set_" + str(setN+1))
		for repN in range(CONTRASTS.size):
			##make sure you are still connected to the RZ2
			if RZ2.get_status == 0:
				raise SystemError, "Hardware connection lost"
			##create a dataset for this orientation
			dset = set_group.create_dataset(str(CONTRASTS[repN]), (3,16,num_samples), dtype = 'f')
			##initialize thread pool to stream data
			#dpool = ThreadPool(processes = 3)
			##set the contrast to zero:
			grating.contrast = 0.0
			grating.draw()
			myWin.flip()
			##trigger the RZ2 to begin recording
			RZ2.send_trig(1)
			##start threads
			#sort_thread = dpool.apply_async(RZ2.stream_data, ("sorted", num_samples, 16, "I32", "int"))
			#spk_thread = dpool.apply_async(RZ2.stream_data, ("spkR", num_samples, 16, "F32", "float"))
			#lfp_thread = dpool.apply_async(RZ2.stream_data, ("lfpR", num_samples, 16, "F32", "float"))
			##pause for the Gray time
			core.wait(GRAY_TIME)
			##make sure this isn't a gray trial
			if CONTRASTS[repN] != -1:
				##adjust the contrast
				grating.contrast = CONTRASTS[repN]
				##draw the stimuli and update the window
				print "Showing contrast " + str(CONTRASTS[repN])
				for frameN in range(num_frames):
					grating.phase = (0.026*frameN, 0)
					grating.draw()
					myWin.flip()
			else:
				##continue to display gray screen
				print "Showing zero contrast control"
				core.wait(DRIFT_TIME)
			##set the contrast to zero:
			grating.contrast = 0.0
			grating.draw()
			myWin.flip()
			##pause for the specified time
			core.wait(GRAY_TIME)
			##now save the data to the hdf5 file
			#sorted spikes
			dset[0,:,:] = RZ2.stream_data("sorted", num_samples, 16, "I32", "int")
			#dset[0,:,:] = sort_thread.get()
			#raw spikes
			dset[1,:,:] = RZ2.stream_data("spkR", num_samples, 16, "F32", "float")
			#dset[1,:,:] = spk_thread.get()
			#raw lfp
			dset[2,:,:] = RZ2.stream_data("lfpR", num_samples, 16, "F32", "float")
			#dset[2,:,:] = lfp_thread.get()
			##clean up
			#dpool.close()
			#dpool.join()

	print "Contrast test complete."
	myWin.close()

def end_session():
	dataFile.close()
	RZ2.stop()
	print "data saved!"