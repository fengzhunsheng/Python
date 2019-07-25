# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:27:55 2019

@author: 冯准生
"""
"""
@brief:simply use wx
     
import wx
app=wx.App()
frame=wx.Frame(None,title="Hello,World")
frame.Show(True)
app.MainLoop()
"""
######################################################
"""
@brief:create my frame

import wx
class Frame1(wx.Frame):
    def __init__(self,parent,title):
        wx.Frame.__init__(self,parent,title=title,pos=(100,200),size=(400,400))
        panel=wx.Panel(self)
        text1=wx.TextCtrl(panel,value="Hello,World!",size=(200,100))
        self.Show(True)

if __name__=='__main__':
    app=wx.App()
    frame=Frame1(None,"Example")
    app.MainLoop()
"""
###########################################################
import wx
class Frame2(wx.Frame):
    def __init__(self,superior):
        wx.Frame.__init__(self,parent=superior,
                          title="Hello,World!",size=(400,400))
        self.panel=wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_UP,self.OnClick)
    def OnClick(self,event):
        posm=event.GetPosition()
        wx.StaticText(parent=self.panel,
                      label="Hello,World!",pos=(posm.x,posm.y))
if __name__=='__main__':
    app=wx.App()
    frame=Frame2(None)
    frame.Show()
    app.MainLoop()







