import tkinter as tk
from AppController import AppController
from CreateGridWidgetHelper import CreateGridWidgetHelper
from CanvasDataClass import CanvasDataClass

class AppView(tk.Tk):
    def __init__(self, controller:AppController):
        super().__init__()

        self.controller = controller
        self.image_view_canvas_data = CanvasDataClass()
        self.title("ImageViewer")
        self.state("zoomed") # フルスクリーン
        # self.geometry("1200x800")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure([0,1], weight=1) 
        self._create_widget()
        self._create_menu()
        self.create_canvas_popup_menu()
        self._bind_canvas_event()

    def _create_menu(self):
        """メニューの生成"""
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="画像選択", command=self.select_image_file)
        file_menu.add_command(label="保存", command=self.save_image)
        file_menu.add_command(label="終了", command=self.exit_app)
        menu_bar.add_cascade(label="メニュー", menu=file_menu)
        self.config(menu=menu_bar)

    def create_canvas_popup_menu(self):
        self.popup_menu = tk.Menu(self, tearoff=False)
        self.popup_menu.add_command(label="保存", command=self.save_image)
        self.popup_menu.add_command(label="リセット", command=self.event_reset_button_click)
        self.popup_menu.add_command(label="グレー化", command=self.event_reset_button_click)
        

    def _create_widget(self):
        """画面内のウィジェット生成"""

        # キャンバス配置
        self.canvas_frame = CreateGridWidgetHelper.tk_frame(self, rowconfigure=[0], columnconfigure=[0])
        self.image_view_canvas = CreateGridWidgetHelper.canvas(self.canvas_frame)
        
        # ボタン配置
        self.button_frame = CreateGridWidgetHelper.tk_frame(self, position=(1,0), rowconfigure=[0,1,2,3,4], columnconfigure=[0], relief=tk.SUNKEN)
        self.cut_image_button = CreateGridWidgetHelper.ttk_button(self.button_frame, text="切り取り", command=self.event_cut_image_button_click, position=(0,0))
        self.change_binary_button = CreateGridWidgetHelper.ttk_button(self.button_frame, text="2値化", command=self.event_change_binary_button_click, position=(0,1))
        self.detect_area_button = CreateGridWidgetHelper.ttk_button(self.button_frame, text="領域検出", command=self.event_detect_area_button_click, position=(0,2))
        self.calucate_mean_button = CreateGridWidgetHelper.ttk_button(self.button_frame, text="平均値計算", command=self.event_calucate_mean_button_click, position=(0,3))
        self.reset_button = CreateGridWidgetHelper.ttk_button(self.button_frame, text="リセット", command=self.event_reset_button_click, position=(0,4))

        # 情報表示用エントリ配置
        self.image_data_frame = CreateGridWidgetHelper.tk_frame(self, position=(0,1), rowconfigure=0, relief=tk.SUNKEN)
        self.image_data_frame.columnconfigure([1,3,5], weight=1)
        self.image_position_entry = CreateGridWidgetHelper.ttk_label_and_entry(self.image_data_frame, label_text="image_xy：", position=(0,0))
        self.image_rgb_entry = CreateGridWidgetHelper.ttk_label_and_entry(self.image_data_frame, label_text="rgb：", position=(2,0) )
        self.image_mean_entry = CreateGridWidgetHelper.ttk_label_and_entry(self.image_data_frame, label_text="image_mean：", position=(4,0))

    def select_image_file(self):
        self.controller.select_image_file(self.image_view_canvas, self.image_view_canvas_data)

    def save_image(self):
        self.controller.save_image()

    def reset_image(self):
        self.controller.refresh_image()

    def gray_image(self):
        self.controller.gray_image(canvas=self.image_view_canvas, canvas_data=self.image_view_canvas_data)

    def exit_app(self):
        self.destroy()


    def _bind_canvas_event(self):
        self.image_view_canvas.bind("<Motion>", self.event_mouse_motion)
        self.image_view_canvas.bind("<Button-1>", self.event_mouse_down)
        self.image_view_canvas.bind("<B1-Motion>", self.event_mouse_dragging)
        self.image_view_canvas.bind("<ButtonRelease-1>", self.event_mouse_release)
        self.image_view_canvas.bind("<Button-3>", self.show_popup_menu)

    def event_mouse_motion(self, event):
        self.controller.event_mouse_motion(event, self.image_view_canvas_data, self.image_position_entry, self.image_rgb_entry)

    def event_mouse_down(self, event):
        self.controller.event_mouse_down(event, self.image_view_canvas_data)
        
    def event_mouse_dragging(self, event):
        self.controller.event_mouse_dragging(event, self.image_view_canvas_data, self.image_view_canvas)
        
    def event_mouse_release(self, event):
        self.controller.event_mouse_release(self.image_view_canvas_data)

    def show_popup_menu(self, event):
        self.popup_menu.post(event.x_root, event.y_root)
       
    def event_cut_image_button_click(self):
        self.controller.event_cut_image_button_click(self.image_view_canvas_data, self.image_view_canvas)
             
    def event_change_binary_button_click(self):
        self.controller.event_change_binary_button_click(self.image_view_canvas_data, self.image_view_canvas)
        
    def event_reset_button_click(self):
        self.controller.event_reset_button_click(self.image_view_canvas_data, self.image_view_canvas)
            
    def event_detect_area_button_click(self):
        self.controller.event_detect_area_button_click(self.image_view_canvas_data, self.image_view_canvas)
           
    def event_calucate_mean_button_click(self):
        self.controller.event_calucate_mean_button_click(self.image_view_canvas_data, self.image_mean_entry)