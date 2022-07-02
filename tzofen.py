from kivy.app import Widget
import kivy
kivy.require('2.1.0') # replace with your current kivy version !
from kivy.app import Builder
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout

class RootWid(GridLayout):
    pass

class TzofenApp(App):

    def build(self):
        return RootWid()


if __name__ == '__main__':
    TzofenApp().run()