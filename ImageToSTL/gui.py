import FreeSimpleGUI as sg
import os

from image_to_stl import image_to_stl

SYMBOL_UP =    '▲'
SYMBOL_DOWN =  '▼'

def collapse(layout, key, pad, starts_visible):
	"""
	Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
	:param layout: The layout for the section
	:param key: Key used to make this seciton visible / invisible
	:return: A pinned column that can be placed directly into your layout
	:rtype: sg.pin
	"""
	return sg.pin(sg.Column(layout, key=key, pad=pad, visible=starts_visible))


def display_status_message(window, message, color='red'):
	window['-STATUS TEXT-'].update(message)
	window['-STATUS TEXT-'].update(visible=True)
	window['-STATUS TEXT-'].update(text_color=color)

def process_gui_values(window, values):
	file_name = values['input_filename']
	if not os.path.isfile(file_name):
		display_status_message(window, "Error: Please enter a valid input file.")
		return

	try:
		base = float(values['base_height'])
	except ValueError:
		display_status_message(window, "Error: Base height must be of type 'float'.")
		return	

	try:
		x_scale = float(values['scale_x'])
		if x_scale <= 0.0:
			display_status_message(window, "Error: X scale must be > 0.")
	except ValueError:
		display_status_message(window, "Error: X scale must be of type 'float'.")
		return	

	try:
		y_scale = float(values['scale_y'])
		if y_scale <= 0.0:
			display_status_message(window, "Error: Y scale must be > 0.")
	except ValueError:
		display_status_message(window, "Error: Y scale must be of type 'float'.")
		return	

	try:
		z_scale = float(values['scale_z'])
		if z_scale <= 0.0:
			display_status_message(window, "Error: Z scale must be > 0.")
	except ValueError:
		display_status_message(window, "Error: Z scale must be of type 'float'.")
		return	

	keep_zeroes = not values['ignore_zeroes']

	output_filename = values['output_filename']
	if not output_filename:
		display_status_message(window, "Error: Please enter a valid output filename.")
		return
	
	display_status_message(window,
		f"Generating mesh...", color='white')

	meshed = image_to_stl(file_name, base, x_scale, y_scale, z_scale, keep_zeroes)
	meshed.save(output_filename)

	display_status_message(window, 
		f"Mesh successfully saved as '{output_filename}'.", color='white')


def main():
	section1 = [[sg.Text("Output to:"), 
					sg.InputText('output.stl', key='output_filename', size=(40,1)),
					sg.FileSaveAs(initial_folder='./')],
				[sg.Text("Base height:"), sg.Input(0.0, key='base_height', size=(10,1))],
				[sg.Text("X scale:"), sg.Input(1.0, key='scale_x', size=(10,1))],
				[sg.Text("Y scale:"), sg.Input(1.0, key='scale_y', size=(10,1))],
				[sg.Text("Z scale:"), sg.Input(1.0, key='scale_z', size=(10,1))],
				[sg.Checkbox('Ignore zero-valued pixels', default=True, key='ignore_zeroes')]]


	# All the stuff inside your window.
	layout = [  [sg.Text('ImageToSTL', size=(10, 1), font=('Arial Bold', 14))],
				[sg.Text('Image file:', font=('Arial', 11), pad=(20,0)), 
					sg.Input(key="input_filename"), sg.FileBrowse(initial_folder='./')],
				[sg.T(SYMBOL_DOWN, enable_events=True, k='-OPEN SEC1-', pad=(20,0)), 
					sg.T('Options', enable_events=True, k='-OPEN SEC1-TEXT', font=('Arial', 11))],
				[collapse(section1, '-SEC1-', pad=(35,0), starts_visible=False)],
				[sg.Text("", k='-STATUS TEXT-', text_color='red', visible=False, font=('Arial', 11), pad=(5,3))],
				[sg.Button('Run', enable_events=True, k='-RUN-'), sg.Button('Cancel')] ]

	# Create the Window
	window = sg.Window('ImageToSTL', layout)

	opened1 = False

	# Event Loop to process "events" and get the "values" of the inputs
	while True:
		event, values = window.read()
		if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
			break

		if event.startswith('-OPEN SEC1-'):
			opened1 = not opened1
			window['-OPEN SEC1-'].update(SYMBOL_DOWN if opened1 else SYMBOL_UP)
			window['-SEC1-'].update(visible=opened1)

		if event.startswith('-RUN-'):
			process_gui_values(window, values)

	window.close()

main()