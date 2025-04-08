"""
This file contains all the GUI elements for the tagger program
"""

import tkinter as tk
import tkintermapview as tkmap
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk



from utils.map_display import *
from utils.data_ops import *
from utils.time_conversion import str_time
from program_data import *

import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

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
        
        self.title = label(self, "Track Information")
        grid(self.title, row = 0, column = 0, columnspan= 2, sticky="w")
        
        # Define static text labels:
        self.static_text = [
            label(self, "Observation: "),
            label(self, "Track ID: "),
            label(self, "Site ID: "),
            label(self, "Track Start: "),
            label(self, "Track End: "),
            label(self, "Duration: "),
            label(self, "No. Detects: "),
            label(self, "Confidence: ")
        ]
        
        for i, static_txt in enumerate(self.static_text):
            grid(static_txt, row = i + 1, column = 0, sticky = "w")
            
        # Define dynamic text labels:
        # The keys are used to identify the data frame columns
        self.dynamic_text = {"obs" : label(self, "N/A"), 
             "id_track": label(self, "N/A"),
             "id_site": label(self, "N/A"), 
             "start": label(self, "N/A"),
             "end": label(self, "N/A"),
             "duration": label(self, "N/A"),
             "detections": label(self, "N/A"),
             "confidence": label(self, "N/A")
             }
        
        for i, dynamic_txt in enumerate(self.dynamic_text.values()):
            grid(dynamic_txt, row = i + 1, column = 1, sticky = "w")
            
        configure_grid(self, row_weights=[1]*9, col_weights=[1,1])

    def refresh(self):
        # Obtain the observation data 
        data = self.parent.current_track
        def obtain_new_text(key):
            # Obtain the current observation number
            if key == "obs":
                current = self.parent.idx_obs
                total = self.parent.num_obs
                return str(current) + " / " + str(total)
            elif key == "start":
                return data["sdate"] + " " + data["stime"]
            elif key == "end":
                return data["ldate"] + " " + data["ltime"]
            else:
                return data[key]
        
        for k, v in self.dynamic_text.items():
            update_label(v, obtain_new_text(k))
                

class TrackTags(AppFrame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        
        self.title = tk.Label(self, text = "Track Tags")
        grid(self.title, row = 0, column = 0, columnspan= 2, sticky="w")
        
        # Tags and tag names:
        tags = ["transit", "loiter", "overnight", "cleanup", "fishing_c",
                "fishing_r", "research", "diving", "repairs", "distress", "other"]
        self.tags = {"transit" : "Transit",
                     "loiter" : "Loiter",
                     "overnight" : "Overnight Loiter",
                     "cleanup" : "Cleanup", 
                     "fishing_c" : "Fishing Comm.",
                     "fishing_r" : "Fishing Rec.",
                     "research" : "Research",
                     "diving" : "Diving",
                     "repairs" : "Repairs",
                     "distress" : "Distress",
                     "other" : "Other"}
        
        # Create variables for Checkboxes
        self.variables = dict.fromkeys(set(tags), tk.IntVar(self, 0))
        for k in tags:
            self.variables[k] = tk.IntVar(self, value = 0)
            
        # Create widgets for Checkboxes
        self.checkboxes = dict.fromkeys(set(tags), None)
        for k in tags:
            self.checkboxes[k] = checkbutton(self, self.tags[k], self.variables[k])
            
        # Render Page Layout:
        height = 6
        for i, k in enumerate(tags):
            r = (i % height) + 1
            if i // height == 0:
                grid(self.checkboxes[k], row = r, column = 0, sticky = "w")
            else: # i // height == 1:
                grid(self.checkboxes[k], row = r, column = 1, sticky = "w", padx = 10)
        
        configure_grid(self, row_weights=[1,1,1,1,1,1,1], col_weights=[1,1])

    def refresh(self):
        # Pull tag data from the current data row
        data = self.parent.current_track
        for k,v in self.variables.items():
            v.set(data[k])
            
    def save(self):
        for k,v in self.variables.items():
            self.parent.current_track[k] = v.get()
        
class TrackNotes(AppFrame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        
        self.title = tk.Label(self, text = "Notes")
        grid(self.title, row = 0, column = 0, columnspan= 2, sticky="w")
        
        # Textbox
        self.box = tk.Text(self, width=50, height = 5)
        grid(self.box, row = 1, column = 0, sticky = "nsew")
        
        # Save Button. 
        self.button = button(self, "Submit Tags", self.submit)
        grid(self.button, row = 2, column = 0, sticky = "nsew")
        
        configure_grid(self, row_weights=[1,3,1], col_weights=[1])
        
    def refresh(self):
        # Pulldata from the current notes
        data = self.parent.current_track
        if pd.isna(data["notes"]):
            note = ""
        else:
            note = data["notes"]
    
        self.box.delete("1.0", tk.END)
        self.box.insert(tk.END, note)
        
    def submit(self):
        # Triggers the application level save function
        self.parent.save()
        
    def save(self):
        # Only save non-empty notes
        note = self.box.get("1.0", tk.END).strip()
        if len(note) > 0:
            self.parent.current_track["notes"] = note
        
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
        grid(self.map_widget, row = 1, column=0, sticky="nsew")
        
        configure_grid(self, row_weights=[0,1], col_weights=[1])
        
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
        self.map_widget.set_path(list(map(tuple, trajectory)), width = 4)
        start_lat, start_long = tuple(trajectory[0])
        end_lat, end_long = tuple(trajectory[-1])
        self.map_widget.set_marker(start_lat, start_long, text = "Start",
                                   marker_color_circle = "dark green", 
                                   marker_color_outside = "forest green")
        self.map_widget.set_marker(end_lat, end_long, text = "End",
                                   marker_color_circle = "red4",
                                   marker_color_outside = "firebrick2")
        
    def get_trajectory_center(self, trajectory):
        """
        Auxillary function that calculate a reasonable center for the trajectory
        """
        return np.mean(np.array(trajectory), axis=0)
    
    def clear_map(self):
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
    
    def refresh(self):
        trajectory = self.parent.current_trajectory
        center = self.get_trajectory_center(trajectory)
        # Clear the current map
        self.clear_map()
        self.center_map(center, self.default_zoom)
        self.add_trajectory(trajectory)
    

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
        if self.detections_path is None or self.track_path is None:
            messagebox.showerror(title="Some Files Are Missing!", message= "Some Required Files Are Missing.")
            return
        
        # Pass the filenames to the parent process.
        self.parent.track_path = self.track_path
        self.parent.detections_path = self.detections_path
        self.parent.file_load_window = None
        self.destroy()
        
class DataWindow(AppWindow):
    def __init__(self, parent, takefocus = True):
        super().__init__(parent, takefocus=takefocus)
        self.title("Data Controls")
        self.resizable(False, False)
        self.geometry("500x100")
        self.configure(background='white')
        
        # Possible names for filter tags
        self.names = ["None", 'Transit', 'Loiter', 'Overnight Loiter', 'Cleanup', 'Fishing Comm.', 'Fishing Rec.', 'Research', 'Diving', 'Repairs', 'Distress', 'Other']
        # This variable should corresponds to the display name
        self.filter_name = tk.StringVar(value="None")
        
        # Render GUI Elements
        self.labels = [
            label(self, "Go to Entry: "),
            label(self, "Set Filter: ")
        ]
        for i, l in enumerate(self.labels):
            grid(l, row = i + 1, column = 0, sticky = "w")
            
        self.controls = [
            button(self, "Prev. Untagged", self.parent.prev_untagged),
            button(self, "Prev Record", self.parent.prev),
            button(self, "Next Record", self.parent.next),
            button(self, "Next Untagged", self.parent.next_untagged)
        ]
        for i, b in enumerate(self.controls):
            grid(b, row = 0, column = i, sticky = "nsew")
            
        self.buttons_sides = [
            button(self, "Go", self.go_to),
            button(self, "Close Window", self.close_window)
        ]
        for i, b in enumerate(self.buttons_sides):
            grid(b, row = i + 1, column = 3, sticky = "nsew")
        
        # Selection Box for Filter
        self.selection = ttk.Combobox(self, textvariable=self.filter_name)
        self.selection["values"] = self.names
        grid(self.selection, row = 2, column = 1, sticky = "nsew", columnspan = 2)
        self.selection.bind("<<ComboboxSelected>>", self.apply_filter)
        
        # Entry Box for Seek
        self.textbox = tk.Entry(self)
        grid(self.textbox, row = 1, column = 1, sticky = "nsew", columnspan = 2)
        
        configure_grid(self, row_weights=[1,1,1,1], col_weights=[1,1,1,1])
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        
    def apply_filter(self, _event):
        """
        Apply a given filter
        """
        # First, lookup the corresponding codename for the filter
        lookup_table = {
            "None" : None,
            "Transit" : "transit",
            "Loiter" : "loiter",
            "Overnight Loiter" : "overnight",
            "Cleanup" : "cleanup",
            "Fishing Comm." : "fishing_c",
            "Fishing Rec." : "fishing_r",
            "Research" : "research",
            "Diving" : "diving",
            "Repairs" : "repairs",
            "Distress" : "distress",
            "Other" : "other"
        }
        filter = lookup_table[self.selection.get()]
        # call main process's apply filter function
        self.parent.apply_filter(filter)
        self.refresh()
        
    def refresh(self):
        # First read the filter information from the parent process
        filter = self.parent.filter
        lookup_table = {"transit" : "Transit",
                     "loiter" : "Loiter",
                     "overnight" : "Overnight Loiter",
                     "cleanup" : "Cleanup", 
                     "fishing_c" : "Fishing Comm.",
                     "fishing_r" : "Fishing Rec.",
                     "research" : "Research",
                     "diving" : "Diving",
                     "repairs" : "Repairs",
                     "distress" : "Distress",
                     "other" : "Other",
                     None : "None"}
        self.filter_name.set(value = lookup_table[filter])
        
    def go_to(self):
        """
        Triggers application level index seeking function
        """
        text_string = self.textbox.get().strip()
        self.textbox.delete(0, tk.END)
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
        self.title("Vessel Activity Tagger by User001256")
        self.resizable(False, False)
        self.geometry("1280x720")
        self.configure(background='black')
        
        # Defining window partition
        configure_grid(self, row_weights=[1,1,1,1], col_weights=[0,1])
        
        # Defining all GUI Components
        self.track_info_frame = TrackInfo(self, background = "white")
        self.track_info_frame.grid(row = 0, column = 0, sticky="nsew", padx=2, pady=2)
        
        self.track_tags_frame = TrackTags(self, background = "white")
        self.track_tags_frame.grid(row = 1, column = 0, sticky="nsew", padx = 2)
        
        self.track_notes_frame = TrackNotes(self, background = "white")
        self.track_notes_frame.grid(row = 2, column = 0, sticky="nsew", padx = 2, pady = 2)
        
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
        self.filter = None
        
        # Information to be added in the refresh function.
        self.current_track = None
        self.current_trajectory = None
        
        # Windows
        self.data_control_window = None
        self.file_load_window = None

        
    # Main Application Routines (May be triggered by application frame events)
        
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
        print("File Import Successful")
        self.refresh()
        
    def open_data_controls(self):
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files First.")
            return
        if self.data_control_window is not None:
            self.data_control_window.focus()
            return
        
        self.data_control_window = DataWindow(self)
        
    def refresh(self):
        """
        Refresh the entire program, including all frames and external windows.
        """
        if self.data is None: return
        self.current_track = self.data.get_track(self.idx_obs)
        self.current_trajectory = self.data.get_trajectory(self.current_track["id_track"])
        
        self.track_info_frame.refresh()
        self.track_tags_frame.refresh()
        self.track_notes_frame.refresh()
        self.track_map_frame.refresh()
        if self.data_control_window is not None:
            self.data_control_window.refresh()
        
    def save(self):
        """
        Save the record data. Triggered by the data submit button
        """
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files Before Submitting.")
            return
        if self.current_track is None:
            return
        self.track_tags_frame.save()
        self.track_notes_frame.save()
        # Propagate the update to the data frame
        self.data.save_track(self.current_track, self.idx_obs)
        
    def save_to_file(self):
        if self.data is None: 
            messagebox.showerror(title = "No Data Loaded.", message = "Please Load Data Files First.")
            return
        
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
        
    def prev(self):
        if self.data is None: return
        self.save()
        self.idx_obs = self.data.prev(self.idx_obs)
        self.refresh()
        
    def next(self):
        if self.data is None: return
        self.save()
        self.idx_obs = self.data.next(self.idx_obs)
        self.refresh()
        
    def prev_untagged(self):
        if self.data is None: return
        self.idx_obs = self.data.prev_untagged(self.idx_obs)
        self.refresh()
        
    def next_untagged(self):
        if self.data is None: return
        self.idx_obs = self.data.next_untagged(self.idx_obs)
        self.refresh()
        
    # Filter Controls:
    def apply_filter(self, filter):
        print(filter)
        if self.data is None: return
        if filter is None:
            self.data.unset_filter()
            self.filter = filter
        else:
            try:
                self.data.set_filter(filter)
            except:
                messagebox.showerror(title = "Empty Filter", message = f"No observations satisfy the filter {self.filter}")
            else:
                self.filter = filter
        self.refresh()
        
    def quit_app(self):
        self.destroy()
        
        
    
def program():
    app = MainApp()
    app.mainloop()
    





if __name__ == "__main__":
    program()
    pass
