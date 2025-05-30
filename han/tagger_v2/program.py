"""
This file contains all the GUI elements for the tagger program
"""

import tkinter as tk
import tkintermapview as tkmap
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk

from utils.data_primitives import *
from utils.scripts.time_conversion import str_time
from support.program_data import *
from utils.constants import *

import pandas as pd
import numpy as np
import time

pd.options.mode.copy_on_write = True
DEBUG_MODE = True

# Default Paths for quick loading
# default_track = "../../data/type_label.csv"
# default_detections = "../../data/detections_radar_smoothed.csv"

default_track = "../../data/tracks_tagged_v6.csv"
default_detections = "../../data/detections_tagged_smoothed.csv"

#########################
###### GUI Helpers ######
#########################

def configure_grid(tk_container, row_weights, col_weights):
    """
    Set up the grid configuration for a tkinter container

    Args:
        tk_container: container
        row_weights: weight parameters in rowconfigure for each row
        col_weights: weight parameters in columnconfigure for each column
    """
    for r, w in enumerate(row_weights):
        tk_container.rowconfigure(r, weight = w)

    for c, w in enumerate(col_weights):
        tk_container.columnconfigure(c, weight = w)

def grid(tk_container, **kwargs):
    """
    Wrapper function for the tkinter .grid method

    Args:
        tk_container: tkinter container
        kwargs: arguments for the .grid method
    """
    tk_container.grid(**kwargs)
    
def button(tk_container, text, click_fn = lambda:0):
    """
    Wrapper function for the tkinter Button class

    Args:
        tk_container: the container to add the button to
        text: button text
        click_fn: function to be called upon click

    Returns:
        Button widget
    """
    return tk.Button(tk_container, text = text, command = click_fn)

def checkbutton(tk_container, text, variable):
    """
    Wrapper function for tk.Checkbutton

    Args:
        tk_container: _description_
        text: _description_
        variable: _description_
    """
    return tk.Checkbutton(tk_container, text = text, variable = variable)

def label(tk_container, text = " "):
    """
    Wrapper function for the tkinter Label class

    Args:
        tk_container: the container to add the text to
        text: label text

    Returns:
        Label widget
    """
    return tk.Label(tk_container, text = text)

def update_label(tk_label, text):
    """
    Update the label text for label, button etc.
    """
    # Automatically converts non-string 
    if not isinstance(text, str):
        text = str(text)
    tk_label.config(text = text)


#################################################
###### Standardized Application Components ######
#################################################

class AppFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        
    #! All application frame must contain a refresh function
    #! Application frames may contain an optional save function for data save
    def refresh(self):
        """
        Required refresh function for all app frames.  
        """
        pass
    
class AppWindow(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(master=parent, **kwargs)
        self.parent = parent
    
    #! All application window must contain a refresh and close_window function
    def refresh(self):
        """
        Required refresh function for all app windows.
        """
        pass
    
    def close_window(self):
        """
        Required close_window function for all app windows.
        Call super().close_window() before returining from close_window.
        """
        pass
        

###############################################
###### Individual Application Components ######
###############################################

class ApplicationControls(AppFrame):
    def __init__(self, main_process, **kwargs):
        super().__init__(main_process, **kwargs)
        
        # Define Control Buttons
        self.buttons = [button(self, "Load Files", self.parent.load_file),
                        button(self, "Data", self.parent.open_data_controls),
                        button(self, "Prev Record", self.parent.prev),
                        button(self, "Next Record", self.parent.next),
                        button(self, "Save As", self.parent.save_to_file),
                        button(self, "Quit App", self.parent.quit_app)
        ]
        # Add buttons to the grid
        for i, b in enumerate(self.buttons):
            grid(b, row = 0, column = i, sticky = "nsew", padx = 10, pady = 10)
            
        configure_grid(self, row_weights=[1], col_weights=[1,1,1,1,1,1])
        
    def refresh(self):
        return

class TrackInfo(AppFrame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)

        # Define static text labels:
        self.static_text = [
            label(self, "Observation: "),
            label(self, "Track ID: "),
            label(self, "Track Start: "),
            label(self, "Track End: "),
            label(self, "No. Detects: "),
            label(self, "Min Speed: "),
            label(self, "Max Speed: "),
            label(self, "Avg Speed: "),
            label(self, "Curviness: "),
            label(self, "Avg Heading: "),
            label(self, "SD Heading:"),
            label(self, "Avg Heading Change: "),
            label(self, "SD Heading Change: "),
            label(self, "Distance Traveled: "),
            label(self, "Max Dist from Orig: "),
            label(self, "Predicted Activity: ")
        ]
        
        for i, static_txt in enumerate(self.static_text):
            grid(static_txt, row = i , column = 0, sticky = "w")
            
        # Define dynamic text labels:
        # The keys are used to identify the data frame columns
        self.dynamic_text = {"obs" : label(self, "N/A"), 
             "id_track": label(self, "N/A"),
             "start": label(self, "N/A"),
             "end": label(self, "N/A"),
             "detections": label(self, "N/A"),
             "min_speed" : label(self, "N/A"), 
             "max_speed": label(self, "N/A"),
             "avg_speed": label(self, "N/A"), 
             "curviness": label(self, "N/A"),
             "heading_mean": label(self, "N/A"),
             "heading_std": label(self, "N/A"),
             "turning_mean": label(self, "N/A"),
             "turning_std": label(self, "N/A"),
             "distance_total" : label(self, "N/A"),
             "distance_o" : label(self, "N/A"),
             "act_pred" : label(self, "N/A")
             }
        
        for i, dynamic_txt in enumerate(self.dynamic_text.values()):
            grid(dynamic_txt, row = i, column = 1, sticky = "w")
            
        self.button = button(self, "Show Model Prediction Details", self.show_more)
        
        grid(self.button, row = 16, column = 0, sticky = "nsew", columnspan = 2)
            
        configure_grid(self, row_weights=[1]*17, col_weights=[1,1])

    def refresh(self):
        # Obtain the observation data 
        data = self.parent.current_track_summ
        def obtain_new_text(key):
            # Obtain the current observation number
            if key == "obs":
                current = self.parent.idx_obs
                total = self.parent.num_obs
                return str(current) + " / " + str(total)
            elif key == "id_track":
                return str(self.parent.current_track_data["id_track"])
            elif key == "start":
                return str_time(data["start_time"])
            elif key == "end":
                return str_time(data["end_time"])
            elif key == "act_pred":
                current = self.parent.idx_obs
                return ACT_CODE_2_NAME[self.parent.data.get_model_prediction(current)[0]]
            else:
                return round(data[key], 5)
        
        for k, v in self.dynamic_text.items():
            update_label(v, obtain_new_text(k))
            
    def show_more(self):
        self.parent.open_model_window()
                

class TrackTags(AppFrame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        
        self.title = tk.Label(self, text = "Track Tags")
        grid(self.title, row = 0, column = 0, sticky="w")
    

        self.valid = tk.IntVar(self, value=0)
        self.check = checkbutton(self, "Valid Track", self.valid)  
        
        # Activity Variable, Storing code
        self.activity = tk.StringVar(self, value="")
        # add the choice buttons for activity choice
        
        self.choices = dict()
        for tag, name in zip(ACT_CODE, ACT_NAMES):
            self.choices[tag] = tk.Radiobutton(self, text = name, 
                                               variable=self.activity,
                                               value = tag)
            
        # Should have 10 tags
        
        # Create selection box for vessel type
        self.type = tk.StringVar(value = "Select Vessel Type")
        self.menu = ttk.Combobox(self, textvariable=self.type)
        self.menu["values"] = ["Select Vessel Type"] + TYPE_NAMES[1:]
            
        # Render Page Layout:
        height = 6
        for i, k in enumerate(ACT_CODE[:-1]):
            r, c = (i % height) + 1, i // height
            grid(self.choices[k], row = r, column = c, sticky = "w", padx=10)
        grid(self.choices[""], row = 6, column = 1, sticky = "w", padx=10)
        grid(self.check, row = 7, column = 0, columnspan = 2, sticky = "nsw")
        grid(self.menu, row = 8, column = 0, columnspan = 2, sticky = "nsew")
        
        configure_grid(self, row_weights=[1]*9, col_weights=[1,1])

    def refresh(self):
        # Pull tag data from the current data row
        data = self.parent.current_track_data
        self.activity.set(data["activity"])
        self.valid.set(data["valid"])
            
        # Pull vessel type data:
        vessel_type = None if pd.isna(data["type_m2_agg"]) or len(data["type_m2_agg"]) == 0 else data["type_m2_agg"]
        self.type.set(TYPE_CODE_2_NAME[vessel_type])
            
    def save(self):
        self.parent.current_track_data["activity"] = self.activity.get()
        self.parent.current_track_data["valid"] = self.valid.get()
        self.parent.current_track_data["type_m2_agg"] = TYPE_NAME_2_CODE[self.type.get()]
        
    def show_legacy(self):
        self.parent.open_legacy_tags()
        
class TrackMap(AppFrame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        self.title = tk.Label(self, text = "Trajectory Map:")
        grid(self.title, row = 0, column=0, sticky="w")
        
        # Map Widget
        self.map_widget = tkmap.TkinterMapView(self,
                                               width=self.winfo_width(), 
                                               height=self.winfo_height(), 
                                               corner_radius=0)
        grid(self.map_widget, row = 1, column=0, sticky="nsew", columnspan = 2)
        
        # Reload button
        self.button = button(self, "Reload Map", self.reload_map)
        grid(self.button, row = 0, column = 1, sticky = "nse")
        
        configure_grid(self, row_weights=[0,1], col_weights=[1,0])
        
        self.default_zoom = 13
        
    # Map Control Functions
    def center_map(self, position, zoom):
        """
        Position should be a list / tuple of [latitude, longitude]
        """
        lat, long = tuple(position)
        self.map_widget.set_position(lat, long)
        self.map_widget.set_zoom(zoom)
        
    def add_trajectory(self, trajectory):
        """
        Overlays a trajectory onto the map. Trajectory should be a 2-D array of
        Shape (N, 2), where each row has a lat long record.
        """
        if len(trajectory) < 2: return
        self.map_widget.set_path(list(map(tuple, trajectory)), width = 4)
        
    def get_trajectory_center(self, trajectory):
        """
        Auxillary function that calculate a reasonable center for the trajectory
        """
        max_latlong = np.max(np.array(trajectory), axis=0)
        min_latlong = np.min(np.array(trajectory), axis=0)
        span = np.max(max_latlong - min_latlong)
        # Set the zoom level
        if span < 1e-3:
            zoom = 18
        elif span < 2e-3:
            zoom = 17
        elif span < 5e-3:
            zoom = 16
        elif span < 1e-2:
            zoom = 15
        elif span < 3e-2:
            zoom = 14
        elif span < 0.1:
            zoom = 13
        elif span < 0.3:
            zoom = 12
        else:
            zoom = 11        
        return 0.5 * (max_latlong + min_latlong), zoom
    
    def clear_map(self):
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
    
    def refresh(self):
        trajectory = self.parent.current_trajectory
        center, zoom = self.get_trajectory_center(trajectory)
        # Clear the current map
        self.clear_map()
        self.center_map(center, zoom)
        self.add_trajectory(trajectory)
        
    def reload_map(self):
        # Address the issue of map markers being stuck in map window
        self.parent.reload_map()
    

class FilePopUp(AppWindow):
    def __init__(self, parent, takefocus = True):
        super().__init__(parent, takefocus=takefocus)
        self.title("Load Files")
        self.resizable(False, False)
        self.geometry("800x100")
        self.configure(background='white')
        
        # Filename Storage:
        self.track_path = self.parent.track_path
        self.detections_path = self.parent.detections_path     
        
        self.protocol("WM_DELETE_WINDOW", self.close_window)  
        
        # Render GUI Elements
        self.labels = [
            label(self, "Track File:"),
            label(self, "Detection File:")
        ]
        for i, l in enumerate(self.labels):
            grid(l, row = i, column = 0, sticky = "w")
            
        self.filenames = [
            label(self, "None Selected"),
            label(self, "None Selected")
        ]
        for i, l in enumerate(self.filenames):
            grid(l, row = i, column = 1, sticky = "w")
            
        self.buttons = [
            button(self, "Browse...", self.open_track),
            button(self, "Browse...", self.open_detections),
            button(self, "Done", self.close_window)
        ]
        for i, b in enumerate(self.buttons):
            grid(b, row = i, column = 2, sticky = "e")
            
        configure_grid(self, row_weights=[1,1,1], col_weights=[0,1,0])
        self.refresh()
        
    def refresh(self):
        if self.track_path is not None:
            update_label(self.filenames[0], self.track_path)
        if self.detections_path is not None:
            update_label(self.filenames[1], self.detections_path)
        
    def open_file(self, prompt = "Open..."):
        filetypes = (
            ("csv files", "*.csv"),
            ("All files", "*.*")
        )
        return fd.askopenfilename(title = prompt, initialdir=".", filetypes=filetypes)
    
    def open_track(self):
        self.track_path = self.open_file("Select Track Data.")
        self.refresh()
        
    def open_detections(self):
        self.detections_path = self.open_file("Select Detection Data.")
        self.refresh()
        
    def close_window(self):
        if not self.detections_path or not self.track_path:
            messagebox.showerror(title="Some Files Are Missing!", message= "Some Required Files Are Missing.")
            return
        
        # Pass the filenames to the parent process.
        self.parent.track_path = self.track_path
        self.parent.detections_path = self.detections_path
        self.parent.file_load_window = None
        self.destroy()
        
class ModelDetail(AppWindow):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        self.title("Model Prediction Scores")
        self.resizable(False, False)
        self.geometry("320x160")
        self.configure(background='white')
        
        # Define static text labels:
        self.static_text = [ label(self, item + ":") for item in ACT_NAMES]
        self.static_text.pop(-1)
        
        for i, static_txt in enumerate(self.static_text):
            grid(static_txt, row = i, column = 0, sticky = "w")
            
            
        # Define dynamic text labels:
        # The keys are used to identify the data frame columns
        self.dynamic_text = {k: label(self, "N/A") for k in ACT_CODE}
        self.dynamic_text.pop("")
        
        for i, k in enumerate(ACT_CODE):
            if len(k) == 0: continue
            grid(self.dynamic_text[k], row = i , column = 1, sticky = "w")
            
        configure_grid(self, row_weights=[1]*len(self.static_text), col_weights=[1,1])
        
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        if self.parent.data is not None and self.parent.current_track_data is not None:
            self.refresh()
        
    def refresh(self):
        # Obtain the observation data 
        encodings = get_label_encodings(self.parent.data.model)
        current = self.parent.idx_obs
        _, predictions = self.parent.data.get_model_prediction(current)
        for index, value in enumerate(predictions):
            code = encodings[index]
            update_label(self.dynamic_text[code], np.round(value, 5))
            
    def close_window(self):
        # Clear the window in main application
        self.parent.model_window = None
        self.destroy()
        
class DataWindow(AppWindow):
    def __init__(self, parent, takefocus = True):
        super().__init__(parent, takefocus=takefocus)
        self.title("Data Controls")
        self.resizable(False, False)
        self.geometry("600x200")
        self.configure(background='white')
        
        # Display Names for activity and type
        self.act_display_name = tk.StringVar(value="All")
        self.type_display_name = tk.StringVar(value="All")
        self.model_display_name = tk.StringVar(value = "All")
        
        # Go To Entry Controls
        self.goto_label = label(self, "Go to Entry:")
        grid(self.goto_label, row = 0, column = 0, sticky = "w")
        
        self.goto_textbox = tk.Entry(self)
        grid(self.goto_textbox, row = 0, column = 1, sticky = "nsew", columnspan = 2)
        
        self.goto_button = button(self, "Go", self.go_to)
        grid(self.goto_button, row = 0, column = 3, sticky = "nsew")
        
        # Data Filtering Texts:
        self.left_labels = [
            label(self, "Data Filters"),
            label(self, "By Activity"),
            label(self, "By Vessel Type"),
            label(self, "By Model Prediction"),
        ]
        
        for i, l in enumerate(self.left_labels):
            grid(l, row = i + 1, column = 0, sticky = "w")
            
        self.counter = label(self, "Nan / Nan")
        grid(self.counter, row = 1, column = 3, sticky = "e")
        
        # Filter Dropdown Menus: 
        self.act_menu = ttk.Combobox(self, textvariable=self.act_display_name)
        self.act_menu["values"] = ["All"] + ACT_NAMES 
        self.act_menu.current(0)
        grid(self.act_menu, row = 2, column = 1, sticky = "nsew", columnspan = 3)
            
        self.type_menu = ttk.Combobox(self, textvariable=self.type_display_name)
        self.type_menu["values"] = ["All"] + TYPE_NAMES[1:]
        self.type_menu.current(0)
        grid(self.type_menu, row = 3, column = 1, sticky = "nsew", columnspan = 3)
        
        self.mod_menu = ttk.Combobox(self, textvariable=self.model_display_name)
        self.mod_menu["values"] = ["All"] + ACT_NAMES[:-1]
        self.mod_menu.current(0)
        grid(self.mod_menu, row = 4, column = 1, sticky = "nsew", columnspan = 3)
        
                    
        
        # Other Filtering Options and Checkboxes    
        self.confidence_text = [
            label(self,"Prediction Low Threshold:"),
            label(self,"Prediction High Threshold:")
        ]
        self.confidence_entry = [tk.Entry(self), tk.Entry(self)]
        for i, (e, t) in enumerate(zip(self.confidence_entry, self.confidence_text)):
            grid(t, row = 5 + i, column = 0, sticky = "w", columnspan = 1)
            grid(e, row = 5 + i, column = 1, sticky = "nsew", columnspan = 3)
            
        # Bottom Buttons 
        self.controls = [
            button(self, "Prev Record", self.parent.prev),
            button(self, "Next Record", self.parent.next),
            button(self, "Clear Filter", self.clear_filter),
            button(self, "Apply Filter", self.apply_filter)
        ]
        for i, b in enumerate(self.controls):
            grid(b, row = 7, column = i, sticky = "nsew")
            
        configure_grid(self, row_weights=[1,1,1,1,1,1,1], col_weights=[1,1,1,1])
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.refresh()
        
    def apply_filter(self, _event = None):
        """
        Do some preliminary checks with user input and then send the
        filtering data to the main application
        """
        # First, query all the filter menus to get the codes for filters
        try:
            act_code = ACT_NAME_2_CODE[self.act_menu.get()]
        except KeyError:
            act_code = None
        try:
            type_code = TYPE_NAME_2_CODE[self.type_menu.get()]
        except KeyError:
            type_code = None
        try:
            model_act_code = ACT_NAME_2_CODE[self.mod_menu.get()]
        except KeyError:
            model_act_code = None
        
        conf_lo = self.confidence_entry[0].get()
        conf_hi = self.confidence_entry[1].get()
        
        # Do some preliminary check with the entered confidence filter values:
        try:
            conf_lo = float(conf_lo) if conf_lo else 0.0
            conf_hi = float(conf_hi) if conf_hi else 1.0            
        except ValueError:
            messagebox.showerror(title="Bad Entry.", message= f"Expects floating point values for confidence.")
            return
            
        if max(conf_lo, conf_hi) > 1.0 or min(conf_lo, conf_hi) < 0.0 or conf_lo >= conf_hi:
            messagebox.showerror(title="Bad Entry.", message= f"Invalid confidence bounds.")
            return
        
        # Parse the filter values as a dictionary and pass it to main application
        filter_parameters = dict()
        filter_parameters["tag"] = act_code
        filter_parameters["type"] = type_code
        filter_parameters["pred"] = model_act_code
        filter_parameters["confidence_low"] = conf_lo
        filter_parameters["confidence_high"] = conf_hi
        
        self.parent.apply_filter(filter_parameters)
        
        self.refresh()
        
    def clear_filter(self, _event = None):
        self.parent.clear_filter()
        self.refresh()
        
        
    def refresh(self):
        # Load the filter configuration from the application
        params = self.parent.filter_parameters
        # Obtain string representations of the parameters
        try:
            self.act_display_name.set(ACT_CODE_2_NAME[params["tag"]])
        except KeyError:
            self.act_display_name.set("All")
        tmp = TYPE_CODE_2_NAME[params["type"]]
        self.type_display_name.set(tmp if len(tmp) > 0 else "All")
        self.confidence_entry[0].delete(0, tk.END)
        self.confidence_entry[1].delete(0, tk.END)
        self.confidence_entry[0].insert(tk.END, str(params["confidence_low"]))
        self.confidence_entry[1].insert(tk.END, str(params["confidence_high"]))
            
        update_label(self.counter, f"{self.parent.filter_num_obs} / {self.parent.num_obs}")

    def go_to(self):
        """
        Triggers application level index seeking function
        """
        text_string = self.goto_textbox.get().strip()
        self.goto_textbox.delete(0, tk.END)
        # Bad Entry
        if not text_string.isnumeric():
            messagebox.showerror(title="Bad Entry.", message= f"Expects a numerical index. Entered '{text_string}'.")
            return
        # Out of bounds
        index = int(text_string)
        if index >= self.parent.num_obs or index < 0:
            messagebox.showerror(title="Bad Entry.", message= f"Entered index '{index}' is out of bounds.")
            return
        self.parent.goto(index)
        
    def close_window(self):
        # Set the parent's data control window to None
        self.parent.data_control_window = None
        self.destroy()
        

        
        
    
##############################
###### Main Application ######
##############################

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vessel Activity Tagger (Ver. 0.2.3a)")
        self.resizable(False, False)
        self.geometry("1280x720")
        self.configure(background='black')
        
        # Defining window partition
        configure_grid(self, row_weights=[1,1,1,1], col_weights=[0,1])
        
        # Defining all GUI Components
        self.track_info_frame = TrackInfo(self, background = "white")
        self.track_info_frame.grid(row = 0, column = 0, sticky="nsew", padx=2, pady=2,
                                   rowspan = 2)
        
        self.track_tags_frame = TrackTags(self, background = "white")
        self.track_tags_frame.grid(row = 2, column = 0, sticky="nsew", padx = 2)
        
        
        self.track_map_frame = TrackMap(self, background = "white")
        self.track_map_frame.grid(row = 0, column = 1, rowspan= 3,
                                  sticky="nsew", padx = 2, pady = 2)
        
        self.controls = ApplicationControls(self, background = "white")
        self.controls.grid(row = 3, column = 0, columnspan=2, sticky="nsew")
        
        # Application Data and observation tracking
        self.track_path = ""
        self.detections_path = ""
        self.data = None
        
        # Used for observation tracking
        self.idx_obs = None
        self.num_obs = None
        
        # Filter Parameters
        self.filter_parameters = FILTER_DEFAULT
        self.filter_num_obs = None
        
        # Information to be added in the refresh function.
        self.current_track_data = None
        self.current_track_summ = None
        self.current_trajectory = None
        
        # Windows
        self.data_control_window = None
        self.file_load_window = None
        self.model_window = None
        
        # Bind Hotkeys:
        self.bind("<Left>", self.prev)
        self.bind("<Right>", self.next)
        
        # Try loading from the default path
        self.load_from_default()
        # self.data.fill_all_summaries()
            
    def load_from_default(self):
        try:
            data = ProgramData(default_track, default_detections)
        except KeyboardInterrupt:
            print("Data cannot be loaded from default path, please manually load the data")
            return
        else:
            self.data = data
            self.track_path = default_track
            self.detections_path = default_detections
            self.idx_obs = 0
            self.num_obs = len(self.data)
            self.filter_num_obs = self.num_obs
            self.refresh()

        
    # Main App functions for spawing subwindows
        
    def load_file(self):
        # Prevent load overwrite when opening the load window:
        old_track_path = self.track_path
        old_detection_path = self.detections_path
        
        if self.file_load_window is not None:
            self.file_load_window.focus()
            return
        
        self.file_load_window = FilePopUp(self)
        self.wait_window(self.file_load_window)
        if self.track_path is None or self.detections_path is None:
            messagebox.showerror(title="No File Loaded", message="File Import Unsuccessful.")
            return
        if old_track_path == self.track_path and old_detection_path == self.detections_path:
            print("No new files loaded.")
            return
        
        
        self.data = ProgramData(self.track_path, self.detections_path)
        self.idx_obs = 0
        self.num_obs = len(self.data)
        self.filter_num_obs = self.num_obs
        print("File Import Successful")
        self.refresh()
        
        
    def open_model_window(self):
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files First.")
            return
        if self.model_window is not None:
            self.model_window.focus()
            return
        
        self.model_window = ModelDetail(self)
        
    def open_data_controls(self):
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files First.")
            return
        if self.data_control_window is not None:
            self.data_control_window.focus()
            return
        
        self.data_control_window = DataWindow(self)
        
    def reload_map(self):
        self.track_map_frame.destroy()
        self.track_map_frame = TrackMap(self, background = "white")
        self.track_map_frame.grid(row = 0, column = 1, rowspan= 3,
                                  sticky="nsew", padx = 2, pady = 2)
        if self.data is not None:
            self.track_map_frame.refresh()
        
    def refresh(self):
        """
        Refresh the entire program, including all frames and external windows.
        """
        if self.data is None: return
        self.current_track_data = self.data.get_track(self.idx_obs)
        self.current_track_summ = self.data.get_summary(self.idx_obs)
        id = self.current_track_data["id_track"]
        
        try:
            self.current_trajectory = self.data.get_trajectory(id)
        except RuntimeWarning:
            # Use some default lat long instead...
            self.current_trajectory = [[37.432, -122.170]]
        
        self.track_info_frame.refresh()
        self.track_tags_frame.refresh()
        self.track_map_frame.refresh()
        if self.data_control_window is not None:
            self.data_control_window.refresh()
        if self.model_window is not None:
            self.model_window.refresh()
            
        # Print out some debugging statistics
        if DEBUG_MODE:
            print(self.current_track_data)
        
    def save(self):
        """
        Save the record data. Triggered by the data submit button
        """
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files Before Submitting.")
            return
        if self.current_track_data is None:
            return
        self.track_tags_frame.save()
        
        # Propagate the update to the data frame. (Copy on write principle)
        self.data.save_track(self.current_track_data, self.idx_obs)
        
    def save_to_file(self):
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files First.")
            return
        
        # First make sure to push any unsaved changes to dataframe
        self.save()
        
        def get_filename():
            """
            Input track should have .../tracks_tagged.csv
            """
            return self.track_path.strip().split("/")[-1]
        
        prompt = "Save modified tags..."
        filename = fd.asksaveasfilename(title = prompt, initialfile = get_filename())
        if len(filename.strip()) == 0:
            messagebox.showerror(title = "No Data Saved.", message = "Save Unsuccessful.")
            return
        self.data.save_to_file(filename)
        print(f"Successfully Saved to {filename}")
        
    # Index Controls:
    def goto(self, index):
        if self.data is None: return
        self.idx_obs = index
        self.refresh()
        
    def prev(self, _event = None):
        if self.data is None: return
        self.save()
        self.idx_obs = self.data.prev(self.idx_obs)
        self.refresh()
        
    def next(self, _event = None):
        if self.data is None: return
        self.save()
        self.idx_obs = self.data.next(self.idx_obs)
        self.refresh()
        
    # Filter Controls:
    def apply_filter(self, filter_parameters):
        if self.data is None: return
        # Attempts to set filter
        if not self.data.set_filter(**filter_parameters):
            messagebox.showerror(title = "Empty", message = "No tracks satisfy selected filters.")
            return
        else:
            # Filter set successfully.
            self.filter_parameters = filter_parameters
            self.filter_num_obs = self.data.get_filtered_count()
        self.refresh()
        
    def clear_filter(self):
        if self.data is None: return
        self.filter_parameters = FILTER_DEFAULT
        self.filter_num_obs = self.num_obs
        self.data.unset_filter()
        
    def quit_app(self):
        self.destroy()
    
    
def launch():
    app = MainApp()
    app.mainloop()
    
if __name__ == "__main__":
    launch()
