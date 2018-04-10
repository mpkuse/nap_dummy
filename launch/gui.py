#!/usr/bin/python
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import subprocess
import os

class MyWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="NAP GUI-Launcher")

        self.set_border_width(10)

        hbox = Gtk.Box( spacing=6 )
        self.add( hbox )

        self.button = Gtk.Button(label="test")
        self.button.connect("clicked", self.tesr)
        hbox.pack_start( self.button, True, True, 0 )

        self.roscore_button = Gtk.Button(label="RosCore")
        self.roscore_button.connect("clicked", self.start_roscore)
        hbox.pack_start( self.roscore_button, True, True, 0 )
        # self.add(self.roscore_button)

        self.button = Gtk.Button(label="turbo i7")
        self.button.connect("clicked", self.turbo_i7)
        hbox.pack_start( self.button, True, True, 0 )

        self.button = Gtk.Button(label="turbo tx2")
        self.button.connect("clicked", self.turbo_tx2)
        hbox.pack_start( self.button, True, True, 0 )

        self.button = Gtk.Button(label="launch bag vio+nap+rviz")
        self.button.connect("clicked", self.launch_bag)
        hbox.pack_start( self.button, True, True, 0 )

    def tesr(self, widget):
        print 'test'

        # subprocess.call(['/bin/bash', '-i', '-c', 'roscore'])
        subprocess.Popen(['/bin/bash', '-i', '-c', 'roscore'])

        print '---'
        print 'env', os.environ['PATH']


    def launch_bag(self, widget):
        print("launch bag")
        #subprocess.Popen(['/bin/bash', '-i', '-c', 'roslaunch', 'nap', 'blackbox4_bag.launch'] )

        # subprocess.Popen(['/bin/bash', '-i', '-c', 'ls /'], shell=True )
        os.system(  'roslaunch nap blackbox4_bag.launch' )
        print 'Done@'

    def start_roscore( self, widget ):
        print('Start Roscore')
        subprocess.Popen(['/bin/bash', '-i', '-c', 'roscore'])
        print( 'Done!' )


    def turbo_i7( self, widget ):
        print( "exec /home/i7-2/Desktop/turbo.sh")
        subprocess.Popen( '/home/i7-2/Desktop/turbo.sh')
        print( 'Done!' )

    def turbo_tx2( self, widget ):
        print( "ssh root@192.168.3.5 /home/nvidia/jetson_clocks.sh")
        os.system( 'ssh root@192.168.3.5 /home/nvidia/jetson_clocks.sh' )
        print( 'Done!' )

win = MyWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
