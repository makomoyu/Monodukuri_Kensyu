import tkinter as tk
from CreateGridWidgetHelper import CreateGridWidgetHelper
from JudgeClass2 import ORBJudge

class AppView(tk.Tk):
    def __init__(self):
        super().__init__()

        # self.image_view_canvas_data = CanvasDataClass()
        self.title("判定システム")
        self.state("zoomed") # フルスクリーン
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure([0,1], weight=1) 
        self._create_widget()

        

    def _create_widget(self):
        """画面内のウィジェット生成"""

        # キャンバス配置
        self.canvas_frame = CreateGridWidgetHelper.tk_frame(self, rowconfigure=[0], columnconfigure=[0])
        self.image_view_canvas = CreateGridWidgetHelper.canvas(self.canvas_frame)
        # 判定結果表示用フレーム
        self.hantei_data_frame = CreateGridWidgetHelper.tk_frame(self, position=(1,0), rowconfigure=[0,1], columnconfigure=[0,1], sticky="news")
        self.result_label = CreateGridWidgetHelper.tk_label(self.hantei_data_frame, text="OK", bg="#0080FF", fg="white", font=("Meiryo", 36, "bold"), width=12, height=3, position=(0,0), columnspan=2,sticky="ew")
        self.score_label = CreateGridWidgetHelper.tk_label(self.hantei_data_frame,text="判定：",bg="#FFFFFF",fg="black",font=("Meiryo", 24, "bold"),width=5,height=3,relief="solid",bd=3, position=(0,1), sticky="ew")
        self.score_result_label = CreateGridWidgetHelper.tk_label(self.hantei_data_frame,text="0.85",bg="#FFFFFF",fg="black",font=("Meiryo", 24, "bold"),width=12,height=3,relief="solid",bd=3, position=(1,1), sticky="ew")
        self.result_image_canvas = CreateGridWidgetHelper.canvas(self.hantei_data_frame, position=(0,2), columnspan=2)


    def update_label(self, result_data:dict):
        hantei_dict = result_data.items()[0]
        score_dict = result_data.items[1]
        self.update_hantei_label(hantei_dict)
        self.update_score_result_label(score_dict)
 
    def update_hantei_label(self, hantei_data_dict:dict):
        key, value = hantei_data_dict.items()
        if value == "OK":
            self.result_label.config(text="OK", bg="#0066FF")
        if value == "NG":
            self.result_label.config(text="NG", bg="#FF0000")
        
    def update_score_result_label(self, score_data_dict:dict):
        key, value = score_data_dict.items()
        self.score_result_label.config(text=f"{value}")


    def execute_hantei_processing(self):
        image = None
        tempalte_image = None

        result = ORBJudge.result(image, tempalte_image)
        self.update_label(result_data=result)









if __name__ == "__main__":
    app = AppView()
    app.mainloop()