from tkinter import *
from chat import get_response, bot_name
from recsystem import Recommend

BG_GRAY="#455158" #color of send box #e9ecf2
BG_COLOR="#6b6d78"  #color of text box
TEXT_COLOR="#000000" #color of text

FONT="Helvetica 16"
FONT_BOLD="Helvetica 18 bold"

class ChatApplication:
    def __init__(self):
        self.window=Tk()
        self._setup_main_window()
        self.last_respond = 0

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("DietBOT")
        self.window.resizable(width=True,height=True)
        self.window.configure(width=600, height=600,bg="#455158")

        #head label 
        head_label=Label(self.window, bg="#455158", fg="#00A2FF",
                            text="WELCOME TO DietBOT!\nTO CONTINUE PLEASE TYPE HELP.", font="Helvetica 14 bold", pady=13)
        head_label.place(relwidth=1)

        #tiny divider
        line=Label(self.window, width=450, bg="#455158")
        line.place(relwidth=1, rely=0.7, relheight=0.012)

        #text widget stored as instance variable
        self.text_widget = Text(self.window, width=20,height=2,bg="#455158", fg="#000000",
                                font=FONT, spacing1=6, spacing2=6, padx=5, pady=5)
                                #check pady

        self.text_widget.place(relheight=0.745, relwidth=0.97, rely=0.1)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scroll bar
        scrollbar=Scrollbar(self.window)
        scrollbar.place(relheight=1, relx=0.97)
        scrollbar.configure(command=self.text_widget.yview)

        #bottom label
        bottom_label=Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        #message entry box
        self.msg_entry=Entry(bottom_label, bg="#FFFFFF", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.92, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg,"YOU")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)

        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL, wrap=WORD, spacing3=2)
        self.text_widget.tag_configure('blue',foreground="#00A2FF", font='Helvetica 16 bold')
        self.text_widget.tag_configure('white', foreground="#FFFFFF", font='Helvetica 16 bold')
        self.text_widget.insert(END, "YOU: ",'blue')
        self.text_widget.insert(END, msg+'\n','white')
        self.text_widget.configure(cursor="arrow", state=DISABLED, wrap=WORD)

        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL, wrap=WORD, spacing3=2)
        self.text_widget.tag_configure('red',foreground="#E65050", font='Helvetica 16 bold')
        self.text_widget.tag_configure('white', foreground="#FFFFFF", font='Helvetica 16 bold')
        self.text_widget.insert(END, "DietBOT: ",'red')
        if self.last_respond!=1:
            self.text_widget.insert(END, get_response(msg)+'\n','white')
        elif self.last_respond==1:
            for i in range(len(msg.split())):
                age = int(msg.split()[0])
                weight = int(msg.split()[1])
                height = int(msg.split()[2])
            self.text_widget.insert(END, Recommend(age,weight,height),'white')
            self.last_respond=0
        if get_response(msg)=="Sure! Type your age, weight and height information please\nExample:21 75 180":
            self.last_respond = 1
        self.text_widget.configure(cursor="arrow", state=DISABLED, wrap=WORD)

        self.text_widget.see(END)

app = ChatApplication()
app.run()