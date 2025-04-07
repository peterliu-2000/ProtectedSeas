"""
This file contains all the GUI elements for the tagger program
"""

import tkinter as tk
import tkintermapview as tkmap
from tkinter import filedialog as fd
from tkinter import messagebox



from utils.map_display import *
from utils.data_ops import *
from utils.time_conversion import str_time
from program_data import *

import pandas as pd
import numpy as np


### Main Application Class

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vessel Activity Tagger by User001256")
        self.resizable(False, False)
        self.geometry("1280x720")
        self.configure(background='black')
        
        # Defining window partition
        self.rowconfigure(0, weight = 1)
        self.rowconfigure(1, weight = 1)
        self.rowconfigure(2, weight = 1)
        self.rowconfigure(3, weight = 1)
        self.columnconfigure(0, weight = 0)
        self.columnconfigure(1, weight = 1)
        
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
        self.track_path = None
        self.detections_path = None
        self.data = None
        self.index = None
        
        # Information to be added in the refresh function.
        self.current_track = None
        self.current_trajectory = None
        
        
    def load_file(self):
        pop_up = LoadFile(self)
        self.wait_window(pop_up)
        if self.track_path is None or self.detections_path is None:
            messagebox.showerror(title="No File Loaded", message="File Import Unsuccessful.")
            return
        self.data = ProgramData(self.track_path, self.detections_path)
        self.index = 0
        print("File Import Successful")
        self.refresh()
        
        
    def refresh(self):
        if self.data is None: return
        self.current_row = self.data.get_track(self.index)
        self.current_trajectory = self.data.get_trajectory(self.current_row["id_track"])
        self.track_info_frame.refresh(self.current_row)
        self.track_tags_frame.refresh(self.current_row)
        self.track_notes_frame.refresh(self.current_row)
        self.track_map_frame.clear_map()
        self.track_map_frame.center_map(get_site_coordinates(self.current_row["id_site"]))
        self.track_map_frame.add_trajectory(self.current_trajectory)
        
    def prev(self):
        if self.data is None: return
        self.index = self.data.prev(self.index)
        self.refresh()
        
    def next(self):
        if self.data is None: return
        self.index = self.data.next(self.index)
        self.refresh()
        
    def prev_untagged(self):
        if self.data is None: return
        self.index = self.data.prev_untagged(self.index)
        self.refresh()
        
    def next_untagged(self):
        if self.data is None: return
        self.index = self.data.next_untagged(self.index)
        self.refresh()
        
        
    def kill(self):
        self.destroy()
        
        
        
        
### Individual GUI Components   

class ApplicationControls(tk.Frame):
    def __init__(self, main_process, **kwargs):
        super().__init__(main_process, **kwargs)
        # Need to save the main process because we might be calling their methods
        self.parent = main_process
        
        # Control buttons:
        self.load = tk.Button(self, text = "Load Files", command = self.parent.load_file)
        self.load.grid(row = 0, column = 0, sticky="nsew", padx=10, pady=10)
        self.filter = tk.Button(self, text = "Filter Tags")
        self.filter.grid(row = 0, column=1, sticky="nsew", padx = 10, pady = 10)
        self.prev_tag = tk.Button(self, text = "Prev. Untagged Record", command = self.parent.prev_untagged)
        self.prev_tag.grid(row = 0, column = 2, sticky="nsew", padx=10, pady=10)
        self.prev = tk.Button(self, text = "Previous Record", command = self.parent.prev)
        self.prev.grid(row = 0, column = 3, sticky="nsew", padx=10, pady=10)
        self.next = tk.Button(self, text = "Next Record", command = self.parent.next)
        self.next.grid(row = 0, column = 4, sticky="nsew", padx=10, pady=10)
        self.next_tag = tk.Button(self, text = "Next Untagged Record", command = self.parent.next_untagged)
        self.next_tag.grid(row = 0, column = 5, sticky="nsew", padx=10, pady=10)
        self.save = tk.Button(self, text = "Save to File")
        self.save.grid(row = 0, column = 6, sticky="nsew", padx=10, pady=10)
        self.quit_app = tk.Button(self, text = "Quit App and Save", command=self.parent.kill)
        self.quit_app.grid(row = 0, column= 7, sticky="nsew", padx=10, pady=10)

        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)
        self.columnconfigure(3, weight = 1)
        self.columnconfigure(4, weight = 1)
        self.columnconfigure(5, weight = 1)
        self.columnconfigure(6, weight = 1)
        self.columnconfigure(7, weight = 1)
        self.rowconfigure(0, weight=1)

class TrackInfo(tk.Frame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        
        # Initialize static text labels
        self.title = tk.Label(self, text = "Track Information").grid(
            row = 0, column = 0, columnspan= 2, sticky="w")
        
        self.track_id_name = tk.Label(self, text = "Track ID:").grid(
            row = 1, column = 0 , sticky="w")
        self.site_id_name = tk.Label(self, text = "Site ID:").grid(
            row = 2, column = 0, sticky="w")
        self.start_time_name = tk.Label(self, text = "Track Start:").grid(
            row = 3, column = 0, sticky = "w")
        self.end_time_name = tk.Label(self, text = "Track End:").grid(
            row = 4, column = 0, sticky = "w")
        self.duration_name = tk.Label(self, text = "Duration:").grid(
            row = 5, column = 0, sticky="w")
        self.detection_name = tk.Label(self, text = "Num Detections:").grid(
            row = 6, column = 0, sticky="w")
        self.confidence_name = tk.Label(self, text = "Confidence:").grid(
            row = 7, column = 0, sticky="w")
        
        # Initialize Dynamic Labels
        self.track_id = tk.Label(self, text = "N/A")
        self.track_id.grid(row = 1, column = 1 , sticky="w")
        self.site_id = tk.Label(self, text = "N/A")
        self.site_id.grid(row = 2, column = 1, sticky="w")
        self.start_time = tk.Label(self, text = "N/A")
        self.start_time.grid(row = 3, column = 1, sticky = "w")
        self.end_time = tk.Label(self, text = "N/A")
        self.end_time.grid(row = 4, column = 1, sticky = "w")
        self.duration = tk.Label(self, text = "N/A")
        self.duration.grid(row = 5, column = 1, sticky="w")
        self.detection = tk.Label(self, text = "N/A")
        self.detection.grid(row = 6, column = 1, sticky="w")
        self.confidence = tk.Label(self, text = "N/A")
        self.confidence.grid(row = 7, column = 1, sticky="w")
        
    def refresh(self, data_row:pd.Series):
        """
        Refresh the track information of this application frame

        Args:
            data_row: A row series for the track data
        """
        self.track_id.config(text = str(data_row["id_track"]))
        self.site_id.config(text = str(data_row["id_site"]))
        self.start_time.config(text = data_row["sdate"] + " " + data_row["stime"])
        self.end_time.config(text = data_row["ldate"] + " " + data_row["ltime"])
        self.duration.config(text = str(data_row["duration"]))
        self.detection.config(text = str(data_row["detections"]))
        self.confidence.config(text = str(round(data_row["confidence"], 6)))
        
class TrackTags(tk.Frame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        
        self.title = tk.Label(self, text = "Track Tags").grid(
            row = 0, column = 0, columnspan= 2, sticky="w")
        
        # Creating the storage variables for each tag
        self.transit = tk.IntVar(self, value = 0)
        self.loiter = tk.IntVar(self, value = 0)
        self.overnight = tk.IntVar(self, value = 0)
        self.cleanup = tk.IntVar(self, value = 0)
        self.fishing_c = tk.IntVar(self, value = 0)
        self.fishing_r = tk.IntVar(self, value = 0)
        self.research = tk.IntVar(self, value = 0)
        self.diving = tk.IntVar(self, value = 0)
        self.repairs = tk.IntVar(self, value = 0)
        self.distress = tk.IntVar(self, value = 0)
        self.other = tk.IntVar(self, value = 0)
        
        # Create the checkboxes for each storage variable:
        self.tra = tk.Checkbutton(self, text = "Transit", variable = self.transit)
        self.loi = tk.Checkbutton(self, text = "Loiter", variable = self.loiter)
        self.ove = tk.Checkbutton(self, text = "Overnight Loiter", variable = self.overnight)
        self.cln = tk.Checkbutton(self, text = "Cleanup", variable = self.cleanup)
        self.fsc = tk.Checkbutton(self, text = "Fishing Comm.", variable = self.fishing_c)
        self.fsr = tk.Checkbutton(self, text = "Fishing Rec.", variable = self.fishing_r)
        self.res = tk.Checkbutton(self, text = "Research", variable = self.research)
        self.div = tk.Checkbutton(self, text = "Diving", variable = self.diving)
        self.rep = tk.Checkbutton(self, text = "Repairs", variable = self.repairs)
        self.dis = tk.Checkbutton(self, text = "Distress", variable = self.distress)
        self.oth = tk.Checkbutton(self, text = "Other", variable = self.other)
        
        self.tra.grid(row = 1, column = 0, sticky="w")
        self.loi.grid(row = 2, column = 0, sticky="w")
        self.ove.grid(row = 3, column = 0, sticky="w")
        self.cln.grid(row = 4, column = 0, sticky="w")
        self.fsc.grid(row = 5, column = 0, sticky="w")
        self.fsr.grid(row = 6, column = 0, sticky="w")
        self.res.grid(row = 1, column = 1, sticky="w", padx = 10)
        self.div.grid(row = 2, column = 1, sticky="w", padx = 10)
        self.rep.grid(row = 3, column = 1, sticky="w", padx = 10)
        self.dis.grid(row = 4, column = 1, sticky="w", padx = 10)
        self.oth.grid(row = 5, column = 1, sticky="w", padx = 10)
        
    def refresh(self, data_row:pd.Series):
        self.transit.set(data_row["transit"])
        self.loiter.set(data_row["loiter"])
        self.overnight.set(data_row["overnight"])
        self.cleanup.set(data_row["cleanup"])
        self.fishing_c.set(data_row["fishing_c"])
        self.fishing_r.set(data_row["fishing_r"])
        self.research.set(data_row["research"])
        self.diving.set(data_row["diving"])
        self.repairs.set(data_row["repairs"])
        self.distress.set(data_row["distress"])
        self.other.set(data_row["other"])
        
    def save_tags(self, data_row:pd.Series):
        data_row["transit"] = self.transit.get()
        data_row["loiter"] = self.loiter.get()
        data_row["overnight"] = self.overnight.get()
        data_row["cleanup"] = self.cleanup.get()
        data_row["fishing_c"] = self.fishing_c.get()
        data_row["fishing_r"] = self.fishing_r.get()
        data_row["research"] = self.research.get()
        data_row["diving"] = self.diving.get()
        data_row["repairs"] = self.repairs.get()
        data_row["distress"] = self.distress.get()
        data_row["other"] = self.other.get()

class TrackNotes(tk.Frame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        self.title = tk.Label(self, text = "Notes")
        self.title.grid(row = 0, column = 0, sticky="w")
        
        # Textbox and entered text
        self.textbox = tk.Text(self, width = 50, height = 5)
        self.textbox.grid(row = 1, column = 0, sticky="nsew")
        
        self.entered_note = ""
        
        # Submit button
        self.button = tk.Button(self, text = "Save Current Record", command = self.submit)
        self.button.grid(row = 2, column=0, sticky="nsew")
        
        # Defining window partition
        self.rowconfigure(0, weight = 1)
        self.rowconfigure(1, weight = 3)
        self.rowconfigure(2, weight = 1)
        
    def refresh(self, data_row:pd.Series):
        """
        Refresh the vessel notes frame

        Args:
            data_row
        """
        if pd.isna(data_row["notes"]):
            self.entered_note = ""
        else:
            self.entered_note = data_row["notes"]
        
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, self.entered_note)
    
    def submit(self, _event = None):
        """
        Submit the edited notes, but not yet saving to the data frame
        """
        text = self.textbox.get("1.0", tk.END).strip()
        if len(text) > 0:
            self.entered_note = text
            print(self.entered_note)
            
    def save_text(self, data_row:pd.Series):
        if len(self.entered_note) > 0:
            data_row["notes"] = self.entered_note
        
class TrackMap(tk.Frame):
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
        self.title = tk.Label(self, text = "Trajectory Map View:")
        self.title.grid(row = 0, column=0, sticky="w")
        self.map_widget = tkmap.TkinterMapView(self,
                                               width=self.winfo_width(), 
                                               height=self.winfo_height(), 
                                               corner_radius=0)
        self.map_widget.grid(row = 1, column=0, sticky="nsew")
        self.rowconfigure(0, weight = 0)
        self.rowconfigure(1, weight = 1)
        self.columnconfigure(0, weight = 1)
    
    def center_map(self, position, zoom = 13):
        lat, long = tuple(position)
        self.map_widget.set_position(lat, long)
        self.map_widget.set_zoom(zoom)
    
    def add_trajectory(self, trajectories):
        self.map_widget.set_path(list(map(tuple, trajectories)), width = 4)
        start_lat, start_long = tuple(trajectories[0])
        end_lat, end_long = tuple(trajectories[-1])
        self.map_widget.set_marker(start_lat, start_long, text = "Start",
                                   marker_color_circle = "dark green", 
                                   marker_color_outside = "forest green")
        self.map_widget.set_marker(end_lat, end_long, text = "End",
                                   marker_color_circle = "red4",
                                   marker_color_outside = "firebrick2")
        
    def clear_map(self):
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
        
        
        
        
### Program Sub Windows
class LoadFile(tk.Toplevel):
    def __init__(self, master, takefocus = True):
        super().__init__(master=master, takefocus=takefocus)
        self.title("Load FIles")
        self.resizable(False, False)
        self.geometry("600x100")
        self.configure(background='white')
        
        # Requires sending information back to the parent process.
        self.parent = master
        
        self.track_text = tk.Label(self, text = "Track File:").grid(
            row = 0, column = 0 , sticky="w")
        self.detection_text = tk.Label(self, text = "Detection File:").grid(
            row = 1, column = 0, sticky="w")
        
        self.track = tk.Label(self, text = "None Selected")
        self.track.grid(row = 0, column = 1 , sticky="w")
        self.detection = tk.Label(self, text = "None Selected")
        self.detection.grid(row = 1, column = 1, sticky="w")
        
        self.track_filename = None
        self.detection_filename = None
        
        self.track_open = tk.Button(self, text = "Browse...", command = self.open_track)
        self.track_open.grid(row = 0, column = 2 , sticky="e")
        self.detection_open = tk.Button(self, text = "Browse...", command = self.open_detections)
        self.detection_open.grid(row = 1, column = 2, sticky="e")
        self.quit_button = tk.Button(self, text = "Done", command = self.close_window)
        self.quit_button.grid(row = 2, column = 2, sticky="e")
        
        self.rowconfigure(0, weight = 1)
        self.rowconfigure(1, weight = 1)
        self.rowconfigure(2, weight = 1)
        self.columnconfigure(0, weight = 0)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 0)
        
        
        
    def open_file(self, prompt = "Open..."):
        filetypes = (
            ("csv files", "*.csv"),
            ("All files", "*.*")
        )
        return fd.askopenfilename(title = prompt, initialdir=".", filetypes=filetypes)
    
    def open_track(self):
        self.track_filename = self.open_file("Select Track Data.")
        if self.track_filename: self.track.config(text = self.track_filename)
        
    def open_detections(self):
        self.detection_filename = self.open_file("Select Detection Data.")
        if self.detection_filename: self.detection.config(text = self.detection_filename)
        
    def close_window(self):
        if self.detection_filename is None or self.track_filename is None:
            messagebox.showerror(title="Some Files Are Missing!", message= "Some Required Files Are Missing.")
            return
        
        # Pass the filenames to the parent process.
        self.parent.track_path = self.track_filename
        self.parent.detections_path = self.detection_filename
        self.destroy()
        
        

        
    
        
        
        
        
        
        
        
        
        
        
        
        

def program():
    app = MainApp()
    app.mainloop()
    





if __name__ == "__main__":
    program()
    pass
