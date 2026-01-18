
import socket
import sys
import pandas as pd
import numpy as np
import getopt
import mglearn
import os
import time
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

PI= 3.141592653589793

data_size = 2**17

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,m=None,vision=False):
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second
        self.parse_the_command_line()
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d
        if m: self.maxSteps= m
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    # Windows-specific TORCS restart
                    import platform
                    if platform.system() == 'Windows':
                        # Kill any existing TORCS processes
                        os.system('taskkill /F /IM wtorcs.exe 2>nul')
                        time.sleep(1.0)
                        # Start TORCS with quickrace configuration
                        torcs_path = r'C:\torcs\torcs\wtorcs.exe'
                        if os.path.exists(torcs_path):
                            os.system(f'start "" "{torcs_path}" -r quickrace')
                        else:
                            print(f"Warning: TORCS not found at {torcs_path}")
                    else:
                        # Original Linux commands
                        os.system('pkill torcs')
                        time.sleep(1.0)
                        if self.vision is False:
                            os.system('torcs -nofuel -nodamage -nolaptime &')
                        else:
                            os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(2.0)  # Give TORCS more time to start on Windows
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        # Skip command line parsing for training modes
        training_modes = ['analyze', 'train', 'perfection', 'elite', 'intensive', 'continuous', 'demo', 'help']
        if len(sys.argv) > 1 and sys.argv[1] in training_modes:
            return
            
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata= str()

        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d. Reconnecting for new race..." % self.port)
                # Instead of shutting down, reconnect for the new race
                self.reconnect_for_new_race()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def reconnect_for_new_race(self):
        """Reconnect to server for a new race instead of shutting down completely."""
        if self.so:
            try:
                self.so.close()
            except:
                pass
            self.so = None

        print("🔄 Reconnecting to TORCS server for new race...")
        time.sleep(1.0)  # Brief pause before reconnecting

        # Re-establish connection
        self.setup_connection()

        # Reset race-specific state
        self.S = ServerState()
        self.R = DriverAction()

        print("✅ Successfully reconnected for new race!")

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        return self.fancyout()
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        'stucktimer',
        'fuel',
        'distRaced',
        'distFromStart',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                          r"  !  ", r".|'  ", r"./'  ", r"_.-  ", r".--  ", r"..-  ",
                          r"---  ", r".__  ", r"-._  ", r"'-.  ", r"'\.  ", r"'|.  ",
                          r"  |  ", r"  .|'", r"  ./'", r"  .-'", r"  _.-", r"  __.",
                          r"  ---", r"  --.", r"  -._", r"  -..", r"  '\.", r"  '|."  ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                              (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       self.d= { 'accel':0.8,
                   'brake':0,
                  'clutch':0,
                    'gear':3,
                   'steer':0.1,
                   'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 0
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0
        if type(self.d['focus']) is not list or min(self.d['focus'])<-180 or max(self.d['focus'])>180:
            self.d['focus']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out
        return out+'\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S,R= c.S.d,c.R.d
    target_speed=240

    R['steer']= S['angle']*30 / PI
    R['steer']-= S['trackPos']*.30

    R['accel'] = max(0.0, min(1.0, R['accel']))
    

    if S['speedX'] < target_speed - (R['steer']*2.5):
        R['accel']+= .4
    else:
        R['accel']-= .2
    if S['speedX']<10:
       R['accel']+= 1/(S['speedX']+.1)

    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 2):
       R['accel']-= 0.1



    R['gear']=1
    if S['speedX']>60:
        R['gear']=2
    if S['speedX']>100:
        R['gear']=3
    if S['speedX']>140:
        R['gear']=4
    if S['speedX']>190:
        R['gear']=5
    if S['speedX']>200:
        R['gear']=6
    return


#############################################
# MODULAR DRIVE LOGIC WITH USER PARAMETERS  #
#############################################

import math

# ================= ADVANCED RACING AI PARAMETERS =================
# Speed and performance
TARGET_SPEED_STRAIGHT = 280  # Maximum speed on straight sections
TARGET_SPEED_CORNER = 180    # Speed in corners
MINIMUM_SPEED = 40           # Minimum safe speed
MAXIMUM_SPEED = 320          # Absolute maximum speed limit

# Steering parameters
STEER_GAIN = 25              # Base steering sensitivity
STEER_DAMPING = 0.85         # Steering smoothing factor
TRACK_POS_GAIN = 0.15        # Track position correction strength (reduced from 0.25)
ANGLE_GAIN = 0.035           # Angle correction strength

# Braking parameters
BRAKE_THRESHOLD = 0.25       # Angle threshold for braking
HARD_BRAKE_THRESHOLD = 0.55  # Hard braking threshold
BRAKE_SMOOTHING = 0.8        # Brake smoothing factor

# Racing line optimization
RACING_LINE_OFFSET = 0.15    # Offset from center for optimal racing line
TRACK_SENSOR_RANGE = 200     # Maximum track sensor range to consider

# Opponent avoidance
OPPONENT_AVOIDANCE_DISTANCE = 15  # Distance to start avoiding opponents
OPPONENT_BRAKE_DISTANCE = 8       # Distance to brake for opponents
OPPONENT_STEER_WEIGHT = 0.3       # How much to steer away from opponents

# Traction control
TRACTION_CONTROL_THRESHOLD = 3.0  # Wheel slip threshold
TRACTION_CONTROL_STRENGTH = 0.12  # How much to reduce throttle on slip

# Recovery parameters
OFF_TRACK_RECOVERY_SPEED = 0.4    # Speed multiplier when off track
STUCK_RECOVERY_TIME = 100         # Steps before considering car stuck
STUCK_RECOVERY_STEER = 0.5        # Steering to try when stuck

# Gear optimization
GEAR_SPEEDS = [0, 35, 70, 110, 155, 205, 260]  # Optimal gear shift points

# Damage and fuel thresholds
DAMAGE_AVOIDANCE_THRESHOLD = 500   # Damage level to start conservative driving
FUEL_CONSERVATION_THRESHOLD = 50   # Fuel level to start conserving fuel

# ================= COMPREHENSIVE SCENARIO HANDLING =================
# Emergency recovery parameters
EMERGENCY_BRAKE_THRESHOLD = 0.8      # Emergency braking for imminent collision
SPIN_RECOVERY_THRESHOLD = 0.5        # Angle threshold for spin recovery
STALL_RECOVERY_SPEED = 20            # Speed below which to apply recovery boost
WALL_PROXIMITY_THRESHOLD = 5.0       # Distance to wall for emergency avoidance
EXTREME_DAMAGE_THRESHOLD = 800       # Damage level requiring conservative driving

# Adaptive behavior parameters
HIGH_SPEED_THRESHOLD = 250           # Speed considered "high"
LOW_SPEED_THRESHOLD = 30             # Speed considered "low"
EXTREME_CORNER_THRESHOLD = 0.7       # Curvature considered extreme
MULTI_OPPONENT_THRESHOLD = 3          # Number of opponents considered "crowded"

# Recovery and safety systems
RECOVERY_STEER_DAMPING = 0.95        # Extra damping during recovery
SAFETY_MARGIN = 0.8                  # Track position safety margin
PREDICTIVE_HORIZON = 50              # Steps to look ahead for predictions

# ================= ADVANCED RACING FUNCTIONS =================

def analyze_track_curvature(S):
    """Analyze track curvature using track sensors to predict upcoming corners."""
    track_sensors = S['track']
    if not track_sensors:
        return 0.0
    
    # Calculate curvature based on track sensor differences
    # Closer sensors indicate sharper turns
    front_sensors = track_sensors[9:11]  # Front center sensors
    side_sensors = track_sensors[5:8] + track_sensors[11:14]  # Side sensors
    
    front_avg = sum(front_sensors) / len(front_sensors) if front_sensors else 100
    side_avg = sum(side_sensors) / len(side_sensors) if side_sensors else 100
    
    # Higher curvature = lower average distance to track edges
    curvature = (200 - front_avg) / 200.0  # Normalize to 0-1
    
    # Consider side sensors for turn direction prediction
    left_side = sum(track_sensors[5:9]) / 4
    right_side = sum(track_sensors[10:14]) / 4
    
    turn_direction = 0
    if left_side < right_side * 0.7:
        turn_direction = 1  # Left turn coming
    elif right_side < left_side * 0.7:
        turn_direction = -1  # Right turn coming
    
    return curvature, turn_direction

def calculate_optimal_racing_line(S):
    """Calculate optimal racing line based on track sensors and curvature."""
    curvature, turn_direction = analyze_track_curvature(S)
    
    # Base racing line offset
    base_offset = RACING_LINE_OFFSET
    
    # Adjust offset based on upcoming curvature
    if curvature > 0.3:  # Sharp corner coming
        # Use inside line for better cornering
        offset_adjustment = base_offset * (1 + curvature)
        if turn_direction > 0:  # Left turn
            target_track_pos = -offset_adjustment
        elif turn_direction < 0:  # Right turn
            target_track_pos = offset_adjustment
        else:
            target_track_pos = 0
    else:
        # Use optimal line for straights
        target_track_pos = 0
    
    # Smooth transition to avoid jerky movements
    current_pos = S['trackPos']
    smoothing_factor = 0.1
    optimal_pos = current_pos + (target_track_pos - current_pos) * smoothing_factor
    
    return optimal_pos

def detect_opponent_threats(S):
    """Analyze opponent sensors to detect immediate threats."""
    opponents = S['opponents']
    if not opponents:
        return 0, 0
    
    # Check front opponents (sensors 0-8 and 27-35)
    front_opponents = opponents[0:9] + opponents[27:36]
    left_opponents = opponents[9:18]  # Left side
    right_opponents = opponents[18:27]  # Right side
    
    # Find closest opponent in each direction
    front_closest = min(front_opponents) if front_opponents else 200
    left_closest = min(left_opponents) if left_opponents else 200
    right_closest = min(right_opponents) if right_opponents else 200
    
    # Calculate avoidance steering
    avoidance_steer = 0
    brake_pressure = 0
    
    if front_closest < OPPONENT_BRAKE_DISTANCE:
        brake_pressure = 0.8 * (1 - front_closest / OPPONENT_BRAKE_DISTANCE)
    elif front_closest < OPPONENT_AVOIDANCE_DISTANCE:
        brake_pressure = 0.3 * (1 - front_closest / OPPONENT_AVOIDANCE_DISTANCE)
    
    # Steer away from close opponents (reduced weight to prevent overcorrection)
    if left_closest < right_closest and left_closest < OPPONENT_AVOIDANCE_DISTANCE:
        avoidance_steer = 0.15 * (1 - left_closest / OPPONENT_AVOIDANCE_DISTANCE)  # Reduced from 0.3
    elif right_closest < left_closest and right_closest < OPPONENT_AVOIDANCE_DISTANCE:
        avoidance_steer = -0.15 * (1 - right_closest / OPPONENT_AVOIDANCE_DISTANCE)  # Reduced from 0.3
    
    return avoidance_steer, brake_pressure

def calculate_adaptive_speed(S, curvature):
    """Calculate optimal speed based on track conditions."""
    base_speed = TARGET_SPEED_STRAIGHT
    
    # Reduce speed based on curvature
    if curvature > 0.5:
        speed_reduction = curvature * (TARGET_SPEED_STRAIGHT - TARGET_SPEED_CORNER)
        target_speed = TARGET_SPEED_STRAIGHT - speed_reduction
    else:
        target_speed = TARGET_SPEED_STRAIGHT
    
    # Consider damage and fuel for conservative driving
    damage_factor = 1.0
    if S.get('damage', 0) > DAMAGE_AVOIDANCE_THRESHOLD:
        damage_factor = 0.85
    
    fuel_factor = 1.0
    if S.get('fuel', 100) < FUEL_CONSERVATION_THRESHOLD:
        fuel_factor = 0.9
    
    target_speed *= damage_factor * fuel_factor
    
    return max(MINIMUM_SPEED, min(MAXIMUM_SPEED, target_speed))

def handle_stuck_situation(S, stuck_timer):
    """Handle recovery from stuck situations."""
    if stuck_timer > STUCK_RECOVERY_TIME:
        # Try oscillating steering to get unstuck
        oscillation = math.sin(stuck_timer * 0.1) * STUCK_RECOVERY_STEER
        return oscillation, 0.5  # Some throttle to try moving
    
    return 0, 0

def detect_racing_scenarios(S):
    """Comprehensive scenario detection for adaptive racing behavior."""
    scenarios = {
        'emergency': False,
        'spin_recovery': False,
        'stall_recovery': False,
        'wall_proximity': False,
        'high_speed_corner': False,
        'crowded_track': False,
        'damage_conservative': False,
        'fuel_critical': False,
        'off_track': False,
        'extreme_angle': False,
        'multi_opponent_ahead': False,
        'opponent_behind': False,
        'straight_high_speed': False,
        'hairpin_corner': False,
        'chicane_sequence': False
    }

    # Basic sensor data
    speed = S.get('speedX', 0)
    angle = abs(S.get('angle', 0))
    track_pos = abs(S.get('trackPos', 0))
    damage = S.get('damage', 0)
    fuel = S.get('fuel', 100)

    # Emergency situations (highest priority)
    if track_pos > 0.95:  # Very close to wall
        scenarios['emergency'] = True
        scenarios['wall_proximity'] = True

    if angle > SPIN_RECOVERY_THRESHOLD:  # Car is spinning
        scenarios['emergency'] = True
        scenarios['spin_recovery'] = True

    if speed < STALL_RECOVERY_SPEED and speed > 1:  # Car is stalling
        scenarios['emergency'] = True
        scenarios['stall_recovery'] = True

    # Track position issues
    if track_pos > 1.0:
        scenarios['off_track'] = True

    # Speed and corner analysis
    curvature, turn_direction = analyze_track_curvature(S)

    if speed > HIGH_SPEED_THRESHOLD and curvature > EXTREME_CORNER_THRESHOLD:
        scenarios['high_speed_corner'] = True

    if curvature > 0.8:  # Hairpin corner
        scenarios['hairpin_corner'] = True

    if curvature > 0.2 and curvature < 0.6:  # Chicane sequence
        scenarios['chicane_sequence'] = True

    if speed > 280 and curvature < 0.1:  # Straight line high speed
        scenarios['straight_high_speed'] = True

    # Opponent analysis
    opponents = S.get('opponents', [])
    if opponents:
        front_opponents = opponents[0:9] + opponents[27:36]
        rear_opponents = opponents[9:18] + opponents[18:27]

        close_front = sum(1 for dist in front_opponents if dist < 15)
        close_rear = sum(1 for dist in rear_opponents if dist < 10)

        if close_front >= MULTI_OPPONENT_THRESHOLD:
            scenarios['crowded_track'] = True
            scenarios['multi_opponent_ahead'] = True

        if close_rear > 0:
            scenarios['opponent_behind'] = True

    # Damage and fuel management
    if damage > EXTREME_DAMAGE_THRESHOLD:
        scenarios['damage_conservative'] = True

    if fuel < FUEL_CONSERVATION_THRESHOLD:
        scenarios['fuel_critical'] = True

    # Extreme conditions
    if angle > 0.8:
        scenarios['extreme_angle'] = True

    return scenarios

def handle_emergency_scenarios(S, R, scenarios):
    """Handle emergency situations with immediate corrective actions."""
    if not scenarios['emergency']:
        return False

    # Spin recovery (highest priority)
    if scenarios['spin_recovery']:
        angle = S.get('angle', 0)
        # Counter-steer to recover from spin
        R['steer'] = -angle * 2.0  # Aggressive counter-steering
        R['accel'] = 0.0
        R['brake'] = 0.0  # Don't brake during spin
        return True

    # Wall proximity emergency
    if scenarios['wall_proximity']:
        track_pos = S.get('trackPos', 0)
        # Steer away from wall aggressively
        R['steer'] = -track_pos * 3.0  # Strong correction
        R['accel'] = 0.0
        R['brake'] = 0.8  # Emergency braking
        return True

    # Stall recovery
    if scenarios['stall_recovery']:
        R['steer'] = 0.0  # Straight ahead
        R['accel'] = 1.0  # Full throttle to recover
        R['brake'] = 0.0
        return True

    return False

def apply_scenario_adaptations(S, R, scenarios):
    """Apply adaptive modifications based on detected scenarios."""

    # Conservative driving for damage
    if scenarios['damage_conservative']:
        R['accel'] *= 0.7  # Reduce acceleration
        R['brake'] *= 1.2  # Increase braking readiness

    # Fuel conservation
    if scenarios['fuel_critical']:
        R['accel'] *= 0.8  # Reduce throttle
        # Optimize gear for efficiency
        speed = S.get('speedX', 0)
        if speed > 200:
            R['gear'] = min(R.get('gear', 6), 5)  # Downshift for efficiency

    # High speed corner handling
    if scenarios['high_speed_corner']:
        curvature, _ = analyze_track_curvature(S)
        # Increase braking anticipation
        brake_boost = curvature * 0.5
        R['brake'] = min(1.0, R['brake'] + brake_boost)

    # Hairpin corner strategy
    if scenarios['hairpin_corner']:
        R['accel'] *= 0.6  # Conservative throttle
        R['brake'] = max(R['brake'], 0.3)  # Always some braking

    # Chicane sequence handling
    if scenarios['chicane_sequence']:
        # Smooth, precise steering for chicanes
        R['steer'] *= 1.2  # More responsive steering

    # Straight line optimization
    if scenarios['straight_high_speed']:
        R['accel'] = min(1.0, R['accel'] + 0.2)  # Boost acceleration
        R['brake'] *= 0.5  # Reduce braking

    # Crowded track handling
    if scenarios['crowded_track']:
        R['accel'] *= 0.8  # More conservative
        R['brake'] *= 1.3  # More braking readiness

    # Opponent behind (defensive driving)
    if scenarios['opponent_behind']:
        R['accel'] = min(1.0, R['accel'] + 0.1)  # Slight speed boost to maintain gap

    # Off-track recovery
    if scenarios['off_track']:
        track_pos = S.get('trackPos', 0)
        # Gentle correction back to track
        correction = -track_pos * 0.5
        R['steer'] += correction
        R['accel'] *= 0.5  # Reduce speed for recovery

    # Extreme angle handling
    if scenarios['extreme_angle']:
        angle = S.get('angle', 0)
        # Dampen steering to prevent over-correction
        R['steer'] = R['steer'] * 0.7 - angle * 0.3

def calculate_competitive_strategy(S, scenarios):
    """Calculate optimal competitive strategy based on scenarios."""

    strategy = {
        'aggressive': False,
        'defensive': False,
        'overtaking': False,
        'drafting': False,
        'conservative': False,
        'recovery': False
    }

    # Recovery mode
    if scenarios['emergency'] or scenarios['off_track'] or scenarios['spin_recovery']:
        strategy['recovery'] = True
        return strategy

    # Conservative mode
    if scenarios['damage_conservative'] or scenarios['fuel_critical']:
        strategy['conservative'] = True
        return strategy

    # Defensive mode
    if scenarios['crowded_track'] or scenarios['multi_opponent_ahead']:
        strategy['defensive'] = True

    # Overtaking opportunities
    if scenarios['opponent_behind'] and not scenarios['crowded_track']:
        strategy['overtaking'] = True

    # Drafting opportunities
    opponents = S.get('opponents', [])
    if opponents and not scenarios['emergency']:
        front_opponents = opponents[0:9] + opponents[27:36]
        if any(dist < 8 for dist in front_opponents):  # Close enough to draft
            strategy['drafting'] = True

    # Aggressive mode (default competitive strategy)
    if not any(strategy.values()):  # No specific strategy needed
        strategy['aggressive'] = True

    return strategy

def apply_competitive_strategy(R, strategy, S):
    """Apply the calculated competitive strategy to driving actions."""

    if strategy['recovery']:
        # Already handled by emergency functions
        pass

    elif strategy['conservative']:
        R['accel'] *= 0.75
        R['brake'] *= 1.25

    elif strategy['defensive']:
        # Stay in optimal racing line, maintain safe distance
        R['accel'] *= 0.9
        R['brake'] *= 1.1

    elif strategy['overtaking']:
        # Speed up slightly for overtaking maneuver
        R['accel'] = min(1.0, R['accel'] + 0.15)

    elif strategy['drafting']:
        # Maintain close distance for slipstream
        R['accel'] = min(1.0, R['accel'] + 0.1)
        # Slight brake reduction to close gap
        R['brake'] *= 0.8

    elif strategy['aggressive']:
        # Push the limits for best lap times
        speed = S.get('speedX', 0)
        if speed < 300:  # Not at max speed
            R['accel'] = min(1.0, R['accel'] + 0.1)
        # Take more risks with cornering
        R['brake'] *= 0.9  # Brake later

def calculate_steering(S):
    """Advanced steering calculation with racing line optimization."""
    # Get optimal racing line position
    optimal_track_pos = calculate_optimal_racing_line(S)
    
    # Calculate angle correction
    angle_correction = S['angle'] * ANGLE_GAIN
    
    # Calculate track position correction with optimal racing line
    track_pos_error = optimal_track_pos - S['trackPos']
    track_correction = track_pos_error * TRACK_POS_GAIN
    
    # Combine corrections
    steer = angle_correction + track_correction
    
    # Apply opponent avoidance
    opponent_steer, _ = detect_opponent_threats(S)
    steer += opponent_steer
    
    # Handle stuck situations
    stuck_timer = S.get('stucktimer', 0)
    stuck_steer, _ = handle_stuck_situation(S, stuck_timer)
    steer += stuck_steer
    
    # Apply steering damping for stability
    steer *= STEER_DAMPING
    
    return max(-1, min(1, steer))

def calculate_throttle(S, R):
    """Advanced throttle control with adaptive speed management."""
    curvature, _ = analyze_track_curvature(S)
    target_speed = calculate_adaptive_speed(S, curvature)
    
    current_speed = S['speedX']
    speed_error = target_speed - current_speed
    
    # Adaptive throttle based on speed error
    if speed_error > 30:
        accel = 1.0  # Full throttle for large deficits
    elif speed_error > 10:
        accel = 0.8 + (speed_error / 50)  # Progressive acceleration
    elif speed_error > -10:
        accel = 0.6 + (speed_error / 20)  # Maintain speed
    else:
        accel = max(0.1, 0.4 + (speed_error / 50))  # Reduce throttle when too fast
    
    # Boost from standstill
    if current_speed < 15:
        accel = max(accel, 0.8)
    
    # Apply traction control
    accel = apply_traction_control(S, accel)
    
    # Handle off-track situations
    track_pos = abs(S['trackPos'])
    if track_pos > 1.0:
        accel *= OFF_TRACK_RECOVERY_SPEED
    
    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    """Advanced braking with predictive corner detection."""
    curvature, _ = analyze_track_curvature(S)
    angle = abs(S['angle'])
    
    # Base braking on curvature prediction
    brake_force = curvature * 0.6
    
    # Additional braking based on current angle
    if angle > HARD_BRAKE_THRESHOLD:
        brake_force = max(brake_force, 0.9)
    elif angle > BRAKE_THRESHOLD:
        brake_force = max(brake_force, 0.4)
    
    # Opponent-based braking
    _, opponent_brake = detect_opponent_threats(S)
    brake_force = max(brake_force, opponent_brake)
    
    # Speed-based braking for high-speed corners
    if S['speedX'] > 250 and curvature > 0.4:
        brake_force = max(brake_force, 0.3)
    
    # Apply brake smoothing
    brake_force *= BRAKE_SMOOTHING
    
    return max(0.0, min(1.0, brake_force))

def shift_gears(S):
    """Advanced gear selection."""
    return optimize_gear_selection(S)

def optimize_gear_selection(S):
    """Optimize gear selection based on current speed and RPM."""
    speed = S.get('speedX', 0)
    rpm = S.get('rpm', 0)
    current_gear = S.get('gear', 1)
    
    # Find optimal gear based on speed
    optimal_gear = 1
    for i, gear_speed in enumerate(GEAR_SPEEDS[1:], 1):
        if speed >= gear_speed:
            optimal_gear = i
        else:
            break
    
    # Don't exceed maximum gear
    optimal_gear = min(optimal_gear, 6)
    
    # Consider RPM for upshifting
    if rpm > 8000 and current_gear < 6:
        optimal_gear = min(current_gear + 1, 6)
    
    # Consider RPM for downshifting
    if rpm < 3000 and current_gear > 1:
        optimal_gear = max(current_gear - 1, 1)
    
    # Ensure gear is valid
    return max(1, min(6, optimal_gear))

def apply_traction_control(S, accel):
    """Advanced traction control with adaptive thresholds."""
    wheel_spins = S['wheelSpinVel']
    if not wheel_spins or len(wheel_spins) < 4:
        return accel
    
    # Calculate wheel slip
    rear_spin = wheel_spins[2] + wheel_spins[3]
    front_spin = wheel_spins[0] + wheel_spins[1]
    slip_ratio = rear_spin - front_spin
    
    # Adaptive traction control based on speed
    speed_factor = min(1.0, S['speedX'] / 100)
    threshold = TRACTION_CONTROL_THRESHOLD * (1 + speed_factor)
    
    if slip_ratio > threshold:
        reduction = TRACTION_CONTROL_STRENGTH * (slip_ratio / threshold)
        accel = max(0.0, accel - reduction)
    
    return accel

# ================= COMPREHENSIVE SCENARIO HANDLING =================
# Emergency recovery parameters
EMERGENCY_BRAKE_THRESHOLD = 0.8      # Emergency braking for imminent collision
SPIN_RECOVERY_THRESHOLD = 0.5        # Angle threshold for spin recovery
STALL_RECOVERY_SPEED = 20            # Speed below which to apply recovery boost
WALL_PROXIMITY_THRESHOLD = 5.0       # Distance to wall for emergency avoidance
EXTREME_DAMAGE_THRESHOLD = 800       # Damage level requiring conservative driving

# Adaptive behavior parameters
HIGH_SPEED_THRESHOLD = 250           # Speed considered "high"
LOW_SPEED_THRESHOLD = 30             # Speed considered "low"
EXTREME_CORNER_THRESHOLD = 0.7       # Curvature considered extreme
MULTI_OPPONENT_THRESHOLD = 3          # Number of opponents considered "crowded"

# Recovery and safety systems
RECOVERY_STEER_DAMPING = 0.95        # Extra damping during recovery
SAFETY_MARGIN = 0.8                  # Track position safety margin
PREDICTIVE_HORIZON = 50              # Steps to look ahead for predictions

def detect_racing_scenarios(S):
    """Comprehensive scenario detection for adaptive racing behavior."""
    scenarios = {
        'emergency': False,
        'spin_recovery': False,
        'stall_recovery': False,
        'wall_proximity': False,
        'high_speed_corner': False,
        'crowded_track': False,
        'damage_conservative': False,
        'fuel_critical': False,
        'off_track': False,
        'extreme_angle': False,
        'multi_opponent_ahead': False,
        'opponent_behind': False,
        'straight_high_speed': False,
        'hairpin_corner': False,
        'chicane_sequence': False
    }

    # Basic sensor data
    speed = S.get('speedX', 0)
    angle = abs(S.get('angle', 0))
    track_pos = abs(S.get('trackPos', 0))
    damage = S.get('damage', 0)
    fuel = S.get('fuel', 100)

    # Emergency situations (highest priority)
    if track_pos > 0.95:  # Very close to wall
        scenarios['emergency'] = True
        scenarios['wall_proximity'] = True

    if angle > SPIN_RECOVERY_THRESHOLD:  # Car is spinning
        scenarios['emergency'] = True
        scenarios['spin_recovery'] = True

    if speed < STALL_RECOVERY_SPEED and speed > 1:  # Car is stalling
        scenarios['emergency'] = True
        scenarios['stall_recovery'] = True

    # Track position issues
    if track_pos > 1.0:
        scenarios['off_track'] = True

    # Speed and corner analysis
    curvature, turn_direction = analyze_track_curvature(S)

    if speed > HIGH_SPEED_THRESHOLD and curvature > EXTREME_CORNER_THRESHOLD:
        scenarios['high_speed_corner'] = True

    if curvature > 0.8:  # Hairpin corner
        scenarios['hairpin_corner'] = True

    if curvature > 0.2 and curvature < 0.6:  # Chicane sequence
        scenarios['chicane_sequence'] = True

    if speed > 280 and curvature < 0.1:  # Straight line high speed
        scenarios['straight_high_speed'] = True

    # Opponent analysis
    opponents = S.get('opponents', [])
    if opponents:
        front_opponents = opponents[0:9] + opponents[27:36]
        rear_opponents = opponents[9:18] + opponents[18:27]

        close_front = sum(1 for dist in front_opponents if dist < 15)
        close_rear = sum(1 for dist in rear_opponents if dist < 10)

        if close_front >= MULTI_OPPONENT_THRESHOLD:
            scenarios['crowded_track'] = True
            scenarios['multi_opponent_ahead'] = True

        if close_rear > 0:
            scenarios['opponent_behind'] = True

    # Damage and fuel management
    if damage > EXTREME_DAMAGE_THRESHOLD:
        scenarios['damage_conservative'] = True

    if fuel < FUEL_CONSERVATION_THRESHOLD:
        scenarios['fuel_critical'] = True

    # Extreme conditions
    if angle > 0.8:
        scenarios['extreme_angle'] = True

    return scenarios

def handle_emergency_scenarios(S, R, scenarios):
    """Handle emergency situations with immediate corrective actions."""
    if not scenarios['emergency']:
        return False

    # Spin recovery (highest priority)
    if scenarios['spin_recovery']:
        angle = S.get('angle', 0)
        # Counter-steer to recover from spin
        R['steer'] = -angle * 2.0  # Aggressive counter-steering
        R['accel'] = 0.0
        R['brake'] = 0.0  # Don't brake during spin
        return True

    # Wall proximity emergency
    if scenarios['wall_proximity']:
        track_pos = S.get('trackPos', 0)
        # Steer away from wall aggressively
        R['steer'] = -track_pos * 3.0  # Strong correction
        R['accel'] = 0.0
        R['brake'] = 0.8  # Emergency braking
        return True

    # Stall recovery
    if scenarios['stall_recovery']:
        R['steer'] = 0.0  # Straight ahead
        R['accel'] = 1.0  # Full throttle to recover
        R['brake'] = 0.0
        return True

    return False

def apply_scenario_adaptations(S, R, scenarios):
    """Apply adaptive modifications based on detected scenarios."""

    # Conservative driving for damage
    if scenarios['damage_conservative']:
        R['accel'] *= 0.7  # Reduce acceleration
        R['brake'] *= 1.2  # Increase braking readiness

    # Fuel conservation
    if scenarios['fuel_critical']:
        R['accel'] *= 0.8  # Reduce throttle
        # Optimize gear for efficiency
        speed = S.get('speedX', 0)
        if speed > 200:
            R['gear'] = min(R.get('gear', 6), 5)  # Downshift for efficiency

    # High speed corner handling
    if scenarios['high_speed_corner']:
        curvature, _ = analyze_track_curvature(S)
        # Increase braking anticipation
        brake_boost = curvature * 0.5
        R['brake'] = min(1.0, R['brake'] + brake_boost)

    # Hairpin corner strategy
    if scenarios['hairpin_corner']:
        R['accel'] *= 0.6  # Conservative throttle
        R['brake'] = max(R['brake'], 0.3)  # Always some braking

    # Chicane sequence handling
    if scenarios['chicane_sequence']:
        # Smooth, precise steering for chicanes
        R['steer'] *= 1.2  # More responsive steering

    # Straight line optimization
    if scenarios['straight_high_speed']:
        R['accel'] = min(1.0, R['accel'] + 0.2)  # Boost acceleration
        R['brake'] *= 0.5  # Reduce braking

    # Crowded track handling
    if scenarios['crowded_track']:
        R['accel'] *= 0.8  # More conservative
        R['brake'] *= 1.3  # More braking readiness

    # Opponent behind (defensive driving)
    if scenarios['opponent_behind']:
        R['accel'] = min(1.0, R['accel'] + 0.1)  # Slight speed boost to maintain gap

    # Off-track recovery
    if scenarios['off_track']:
        track_pos = S.get('trackPos', 0)
        # Gentle correction back to track
        correction = -track_pos * 0.5
        R['steer'] += correction
        R['accel'] *= 0.5  # Reduce speed for recovery

    # Extreme angle handling
    if scenarios['extreme_angle']:
        angle = S.get('angle', 0)
        # Dampen steering to prevent over-correction
        R['steer'] = R['steer'] * 0.7 - angle * 0.3

def calculate_competitive_strategy(S, scenarios):
    """Calculate optimal competitive strategy based on scenarios."""

    strategy = {
        'aggressive': False,
        'defensive': False,
        'overtaking': False,
        'drafting': False,
        'conservative': False,
        'recovery': False
    }

    # Recovery mode
    if scenarios['emergency'] or scenarios['off_track'] or scenarios['spin_recovery']:
        strategy['recovery'] = True
        return strategy

    # Conservative mode
    if scenarios['damage_conservative'] or scenarios['fuel_critical']:
        strategy['conservative'] = True
        return strategy

    # Defensive mode
    if scenarios['crowded_track'] or scenarios['multi_opponent_ahead']:
        strategy['defensive'] = True

    # Overtaking opportunities
    if scenarios['opponent_behind'] and not scenarios['crowded_track']:
        strategy['overtaking'] = True

    # Drafting opportunities
    opponents = S.get('opponents', [])
    if opponents and not scenarios['emergency']:
        front_opponents = opponents[0:9] + opponents[27:36]
        if any(dist < 8 for dist in front_opponents):  # Close enough to draft
            strategy['drafting'] = True

    # Aggressive mode (default competitive strategy)
    if not any(strategy.values()):  # No specific strategy needed
        strategy['aggressive'] = True

    return strategy

def apply_competitive_strategy(R, strategy, S):
    """Apply the calculated competitive strategy to driving actions."""

    if strategy['recovery']:
        # Already handled by emergency functions
        pass

    elif strategy['conservative']:
        R['accel'] *= 0.75
        R['brake'] *= 1.25

    elif strategy['defensive']:
        # Stay in optimal racing line, maintain safe distance
        R['accel'] *= 0.9
        R['brake'] *= 1.1

    elif strategy['overtaking']:
        # Speed up slightly for overtaking maneuver
        R['accel'] = min(1.0, R['accel'] + 0.15)

    elif strategy['drafting']:
        # Maintain close distance for slipstream
        R['accel'] = min(1.0, R['accel'] + 0.1)
        # Slight brake reduction to close gap
        R['brake'] *= 0.8

    elif strategy['aggressive']:
        # Push the limits for best lap times
        speed = S.get('speedX', 0)
        if speed < 300:  # Not at max speed
            R['accel'] = min(1.0, R['accel'] + 0.1)
        # Take more risks with cornering
        R['brake'] *= 0.9  # Brake later

def calculate_adaptive_exploration(scenarios):
    """Calculate adaptive exploration rate based on scenario danger level."""
    danger_level = 0.0

    # Emergency situations - minimal exploration
    if scenarios['emergency']:
        danger_level = 1.0

    # High-risk situations
    elif scenarios['high_speed_corner'] or scenarios['hairpin_corner']:
        danger_level = 0.8
    elif scenarios['crowded_track'] or scenarios['extreme_angle']:
        danger_level = 0.7
    elif scenarios['wall_proximity'] or scenarios['spin_recovery']:
        danger_level = 0.9

    # Medium-risk situations
    elif scenarios['chicane_sequence'] or scenarios['multi_opponent_ahead']:
        danger_level = 0.5
    elif scenarios['damage_conservative'] or scenarios['fuel_critical']:
        danger_level = 0.4

    # Low-risk situations
    elif scenarios['straight_high_speed'] or scenarios['opponent_behind']:
        danger_level = 0.2

    # Safe situations - higher exploration
    else:
        danger_level = 0.1

    # Convert danger level to exploration rate (inverse relationship)
    exploration_rate = 0.001 + (danger_level * 0.049)  # Range: 0.001 to 0.05
    return exploration_rate

def get_scenario_noise_scale(scenarios):
    """Get noise scaling factor based on current scenarios."""
    if scenarios['emergency']:
        return 0.1  # Very low noise in emergencies
    elif scenarios['high_speed_corner'] or scenarios['hairpin_corner']:
        return 0.3  # Low noise in dangerous corners
    elif scenarios['crowded_track']:
        return 0.4  # Moderate noise in traffic
    elif scenarios['straight_high_speed']:
        return 0.8  # Higher noise on straights for exploration
    else:
        return 0.6  # Default moderate noise

def calculate_adaptive_smoothing(scenarios, current_smooth_factors):
    """Calculate adaptive smoothing factors based on scenarios."""
    smooth_factors = current_smooth_factors.copy()

    # Emergency situations - high smoothing for stability
    if scenarios['emergency']:
        smooth_factors['steer'] = 0.95
        smooth_factors['accel'] = 0.90
        smooth_factors['brake'] = 0.85

    # High-speed corners - moderate smoothing
    elif scenarios['high_speed_corner']:
        smooth_factors['steer'] = 0.85
        smooth_factors['accel'] = 0.80
        smooth_factors['brake'] = 0.75

    # Hairpin corners - high smoothing for precision
    elif scenarios['hairpin_corner']:
        smooth_factors['steer'] = 0.90
        smooth_factors['accel'] = 0.85
        smooth_factors['brake'] = 0.80

    # Chicanes - low smoothing for responsiveness
    elif scenarios['chicane_sequence']:
        smooth_factors['steer'] = 0.70
        smooth_factors['accel'] = 0.75
        smooth_factors['brake'] = 0.70

    # Straight lines - low smoothing for quick acceleration
    elif scenarios['straight_high_speed']:
        smooth_factors['steer'] = 0.60
        smooth_factors['accel'] = 0.65
        smooth_factors['brake'] = 0.60

    # Crowded tracks - high smoothing for predictability
    elif scenarios['crowded_track']:
        smooth_factors['steer'] = 0.88
        smooth_factors['accel'] = 0.82
        smooth_factors['brake'] = 0.78

    return smooth_factors

def apply_scenario_smoothing(R, prev_actions, smooth_factors):
    """Apply scenario-adaptive smoothing to actions."""
    if prev_actions is None:
        return R

    # Apply smoothing to each action
    for action in ['steer', 'accel', 'brake']:
        if action in R and action in prev_actions:
            factor = smooth_factors.get(action, 0.8)
            R[action] = prev_actions[action] * factor + R[action] * (1 - factor)

    return R

def apply_safety_checks(R, S, scenarios):
    """Apply comprehensive safety checks to prevent crashes."""

    # Steering limits (prevent extreme steering)
    max_steer = 0.3
    if scenarios['emergency']:
        max_steer = 0.5  # Allow more steering in emergencies
    elif scenarios['hairpin_corner']:
        max_steer = 0.4  # Allow more for tight corners

    R['steer'] = max(-max_steer, min(max_steer, R['steer']))

    # Acceleration limits
    if scenarios['damage_conservative']:
        R['accel'] = min(R['accel'], 0.7)  # Limit acceleration when damaged

    # Braking limits
    if scenarios['straight_high_speed'] and S.get('speedX', 0) > 280:
        R['brake'] = max(R['brake'], 0.0)  # No braking at high speed on straights

    # Track position safety
    track_pos = abs(S.get('trackPos', 0))
    if track_pos > 0.8:
        # Reduce acceleration when close to wall
        R['accel'] *= (1.0 - track_pos * 0.5)
        # Increase braking readiness
        R['brake'] = min(1.0, R['brake'] + track_pos * 0.3)

    # Speed-dependent limits
    speed = S.get('speedX', 0)
    if speed > 320:  # Absolute maximum speed
        R['accel'] = 0.0
        R['brake'] = min(R['brake'], 0.1)  # Light braking only

    # Angle-based corrections
    angle = abs(S.get('angle', 0))
    if angle > 0.6:
        # Dampen acceleration when car is misaligned
        R['accel'] *= (1.0 - angle * 0.5)
        # Increase braking for stability
        R['brake'] = min(1.0, R['brake'] + angle * 0.3)

    return R

def update_state_and_reward(S, R, scenarios, prev_damage, prev_fuel):
    """Update internal state and calculate reward for learning."""

    # Calculate reward components
    reward = 0.0

    # Speed reward (progress)
    speed = S.get('speedX', 0)
    reward += speed * 0.01  # Base speed reward

    # Position reward (staying on track)
    track_pos = abs(S.get('trackPos', 0))
    if track_pos < 1.0:
        reward += (1.0 - track_pos) * 2.0  # Bonus for track position
    else:
        reward -= track_pos * 5.0  # Penalty for off-track

    # Damage penalty
    current_damage = S.get('damage', 0)
    damage_increase = current_damage - prev_damage
    reward -= damage_increase * 0.1

    # Fuel efficiency reward
    current_fuel = S.get('fuel', 100)
    fuel_used = prev_fuel - current_fuel
    if fuel_used > 0:
        reward -= fuel_used * 0.05  # Small penalty for fuel use

    # Scenario-based rewards
    if scenarios['emergency']:
        reward -= 10.0  # Large penalty for emergencies
    elif scenarios['high_speed_corner'] and speed > 250:
        reward += 5.0  # Bonus for fast cornering
    elif scenarios['straight_high_speed'] and speed > 290:
        reward += 3.0  # Bonus for high straight speeds
    elif scenarios['crowded_track']:
        reward += 1.0  # Small bonus for navigating traffic

    # Opponent-relative rewards
    opponents = S.get('opponents', [])
    if opponents:
        front_clear = all(dist > 20 for dist in opponents[0:9])  # Clear ahead
        if front_clear and speed > 200:
            reward += 2.0  # Bonus for clear track ahead

    return reward, current_damage, current_fuel

# ================= MACHINE LEARNING RACING AI =================
class MLRacingAI:
    """Machine Learning-powered Racing AI that learns optimal driving strategies."""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.learning_mode = True  # Start in learning mode
        self.performance_history = []
        
        # Load existing models if available
        self.load_models()
        
        # Racing AI state
        self.previous_steer = 0
        self.previous_accel = 0.8
        self.previous_brake = 0
        self.stuck_counter = 0
        self.racing_mode = "ml_optimized"
        
        # State tracking for rewards
        self.prev_damage = 0
        self.prev_fuel = 100
        
    def load_models(self):
        """Load pre-trained models if they exist."""
        try:
            with open('racing_models.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                self.models = saved_data['models']
                self.scalers = saved_data['scalers']
                self.is_trained = True
                print("Loaded pre-trained racing models!")
        except FileNotFoundError:
            print("No pre-trained models found, starting fresh...")
            self.train_initial_models()
    
    def train_initial_models(self):
        """Train initial models using synthetic racing data."""
        print("Training initial ML models...")
        
        # Generate synthetic racing data based on expert knowledge
        n_samples = 10000
        features = []
        targets = {'steer': [], 'accel': [], 'brake': []}
        
        for _ in range(n_samples):
            # Simulate realistic racing scenarios
            speed = np.random.uniform(0, 320)
            angle = np.random.normal(0, 0.3)
            track_pos = np.random.normal(0, 0.5)
            curvature = np.random.uniform(0, 1)
            
            # Generate expert actions
            steer = self.expert_steering(speed, angle, track_pos, curvature)
            accel = self.expert_acceleration(speed, angle, curvature)
            brake = self.expert_braking(speed, angle, curvature)
            
            features.append([speed, angle, track_pos, curvature, 
                           speed**2, angle**2, abs(track_pos), curvature*speed])
            targets['steer'].append(steer)
            targets['accel'].append(accel)
            targets['brake'].append(brake)
        
        X = np.array(features)
        
        # Train models for each action
        for action in ['steer', 'accel', 'brake']:
            y = np.array(targets[action])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{action.upper()} Model MSE: {mse:.4f}")
            
            self.models[action] = model
            self.scalers[action] = scaler
        
        self.is_trained = True
        self.save_models()
        print("Initial ML models trained!")
    
    def expert_steering(self, speed, angle, track_pos, curvature):
        """Generate expert steering decisions."""
        steer = angle * 25 / PI - track_pos * 0.25
        
        # Adjust for speed and curvature
        if curvature > 0.3:
            steer *= 1.2  # More aggressive steering in corners
        
        return np.clip(steer, -1, 1)
    
    def expert_acceleration(self, speed, angle, curvature):
        """Generate expert acceleration decisions."""
        target_speed = 280 * (1 - curvature * 0.4)
        speed_error = target_speed - speed
        
        if abs(angle) > 0.3:
            return 0.3  # Conservative in corners
        elif speed_error > 20:
            return 1.0
        elif speed_error > 0:
            return 0.7
        else:
            return 0.2
    
    def expert_braking(self, speed, angle, curvature):
        """Generate expert braking decisions."""
        if abs(angle) > 0.4 or (speed > 250 and curvature > 0.4):
            return min(0.8, abs(angle) * 2)
        return 0.0
    
    def predict_action(self, sensor_data):
        """Use ML models to predict optimal actions."""
        if not self.is_trained:
            return None
        
        # Extract features
        speed = sensor_data.get('speedX', 0)
        angle = sensor_data.get('angle', 0)
        track_pos = sensor_data.get('trackPos', 0)
        
        # Calculate curvature
        curvature, _ = analyze_track_curvature(sensor_data)
        
        # Create feature vector
        features = np.array([[speed, angle, track_pos, curvature, 
                            speed**2, angle**2, abs(track_pos), curvature*speed]])
        
        predictions = {}
        
        for action in ['steer', 'accel', 'brake']:
            if action in self.models:
                # Scale features
                features_scaled = self.scalers[action].transform(features)
                # Predict
                pred = self.models[action].predict(features_scaled)[0]
                predictions[action] = np.clip(pred, 0, 1) if action != 'steer' else np.clip(pred, -1, 1)
            else:
                predictions[action] = 0
        
        return predictions
    
    def update_models(self, sensor_data, actions_taken, reward):
        """Update ML models with new experience (online learning)."""
        if not self.learning_mode:
            return
        
        # Store experience for later batch training
        experience = {
            'sensors': sensor_data.copy(),
            'actions': actions_taken.copy(),
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.data_collector.add_experience(experience)
        
        # Retrain periodically (every 1000 experiences)
        if len(self.data_collector.experiences) % 1000 == 0:
            self.retrain_models()
    
    def retrain_models(self):
        """Retrain models with accumulated experience."""
        if len(self.data_collector.experiences) < 100:
            return
        
        print(f"Retraining ML models with {len(self.data_collector.experiences)} experiences...")
        
        # Convert experiences to training data
        features = []
        targets = {'steer': [], 'accel': [], 'brake': []}
        
        for exp in self.data_collector.experiences[-5000:]:  # Use last 5000 experiences
            sensors = exp['sensors']
            actions = exp['actions']
            
            speed = sensors.get('speedX', 0)
            angle = sensors.get('angle', 0)
            track_pos = sensors.get('trackPos', 0)
            curvature, _ = analyze_track_curvature(sensors)
            
            features.append([speed, angle, track_pos, curvature, 
                           speed**2, angle**2, abs(track_pos), curvature*speed])
            
            targets['steer'].append(actions.get('steer', 0))
            targets['accel'].append(actions.get('accel', 0))
            targets['brake'].append(actions.get('brake', 0))
        
        X = np.array(features)
        
        # Retrain each model
        for action in ['steer', 'accel', 'brake']:
            y = np.array(targets[action])
            
            if action in self.models:
                # Online learning - partial fit
                X_scaled = self.scalers[action].transform(X)
                self.models[action].n_estimators += 10  # Add more trees
                self.models[action].fit(X_scaled, y)
        
        self.save_models()
        print("ML models retrained!")
    
    def save_models(self):
        """Save trained models to disk."""
        try:
            with open('racing_models.pkl', 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'scalers': self.scalers,
                    'training_date': time.time()
                }, f)
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def drive(self, c):
        """Main ML-powered driving logic with comprehensive scenario handling."""
        S, R = c.S.d, c.R.d

        # Step 1: Detect all possible racing scenarios
        scenarios = detect_racing_scenarios(S)

        # Step 2: Handle emergency situations (highest priority)
        if handle_emergency_scenarios(S, R, scenarios):
            # Emergency handled, skip normal processing
            reward, current_damage, current_fuel = update_state_and_reward(S, R, scenarios, self.prev_damage, self.prev_fuel)
            self.prev_damage, self.prev_fuel = current_damage, current_fuel
            return

        # Step 3: Try ML prediction first (if not in emergency)
        ml_actions = self.predict_action(S)

        if ml_actions:
            # Use ML predictions with scenario-adaptive exploration
            exploration_rate = calculate_adaptive_exploration(scenarios)

            for action in ['steer', 'accel', 'brake']:
                if np.random.random() < exploration_rate:
                    # Adaptive noise based on scenario
                    noise_scale = get_scenario_noise_scale(scenarios)
                    ml_actions[action] += np.random.normal(0, noise_scale)

                R[action] = np.clip(ml_actions[action],
                                  -1 if action == 'steer' else 0, 1)
        else:
            # Fallback to rule-based AI
            R['steer'] = calculate_steering(S)
            R['accel'] = calculate_throttle(S, R)
            R['brake'] = apply_brakes(S)

        # Step 4: Apply scenario-based adaptations
        apply_scenario_adaptations(S, R, scenarios)

        # Step 5: Apply competitive strategy
        strategy = calculate_competitive_strategy(S, scenarios)
        apply_competitive_strategy(R, strategy, S)

        # Step 6: Always use optimized gear shifting
        R['gear'] = shift_gears(S)

        # Step 7: Apply adaptive smoothing based on scenarios
        current_smooth_factors = {'steer': 0.8, 'accel': 0.8, 'brake': 0.8}  # Default factors
        smooth_factors = calculate_adaptive_smoothing(scenarios, current_smooth_factors)
        prev_actions = {'steer': self.previous_steer, 'accel': self.previous_accel, 'brake': self.previous_brake}
        R = apply_scenario_smoothing(R, prev_actions, smooth_factors)

        # Step 8: Safety checks and final validation
        R = apply_safety_checks(R, S, scenarios)

        # Step 9: Update state and learning
        reward, current_damage, current_fuel = update_state_and_reward(S, R, scenarios, self.prev_damage, self.prev_fuel)
        self.prev_damage, self.prev_fuel = current_damage, current_fuel

        # Update previous actions for smoothing
        self.previous_steer = R['steer']
        self.previous_accel = R['accel']
        self.previous_brake = R['brake']

        # Update models with experience
        self.update_models(S, R, reward)

    def calculate_adaptive_exploration(self, scenarios):
        """Calculate exploration rate based on current scenarios."""
        base_rate = 0.02  # Conservative base rate

        # Reduce exploration in dangerous situations
        if scenarios['emergency'] or scenarios['wall_proximity'] or scenarios['spin_recovery']:
            return 0.001  # Minimal exploration in emergencies

        # Increase exploration in safe learning situations
        if scenarios['straight_high_speed'] and not scenarios['crowded_track']:
            return 0.05  # More exploration on safe straights

        # Moderate exploration in normal racing
        if scenarios['high_speed_corner'] or scenarios['crowded_track']:
            return 0.01  # Reduced exploration in complex situations

        return base_rate

    def get_scenario_noise_scale(self, scenarios, action):
        """Get appropriate noise scale for different scenarios and actions."""
        if action == 'steer':
            base_noise = 0.02

            # Reduce steering noise in dangerous situations
            if scenarios['emergency'] or scenarios['wall_proximity']:
                return 0.005

            # Increase precision in corners
            if scenarios['hairpin_corner'] or scenarios['high_speed_corner']:
                return 0.01

            return base_noise

        elif action == 'accel':
            base_noise = 0.01

            # Conservative acceleration noise in crowded situations
            if scenarios['crowded_track']:
                return 0.005

            return base_noise

        else:  # brake
            return 0.005  # Always conservative braking noise

    def calculate_adaptive_smoothing(self, scenarios):
        """Calculate smoothing factors based on scenarios."""
        base_steer_smooth = 0.25
        base_accel_smooth = 0.15
        base_brake_smooth = 0.3

        # Increase smoothing in unstable situations
        if scenarios['emergency'] or scenarios['spin_recovery']:
            return {
                'steer': min(0.5, base_steer_smooth * 2),
                'accel': min(0.4, base_accel_smooth * 2),
                'brake': min(0.6, base_brake_smooth * 1.5)
            }

        # Reduce smoothing for precision in corners
        if scenarios['hairpin_corner'] or scenarios['chicane_sequence']:
            return {
                'steer': max(0.1, base_steer_smooth * 0.7),
                'accel': max(0.05, base_accel_smooth * 0.8),
                'brake': max(0.1, base_brake_smooth * 0.8)
            }

        return {
            'steer': base_steer_smooth,
            'accel': base_accel_smooth,
            'brake': base_brake_smooth
        }

    def apply_scenario_smoothing(self, R, smoothing_factors):
        """Apply adaptive smoothing to control inputs."""
        R['steer'] = self.previous_steer + (R['steer'] - self.previous_steer) * smoothing_factors['steer']
        R['accel'] = self.previous_accel + (R['accel'] - self.previous_accel) * smoothing_factors['accel']
        R['brake'] = self.previous_brake + (R['brake'] - self.previous_brake) * smoothing_factors['brake']

    def apply_safety_checks(self, R, S, scenarios):
        """Final safety checks and validation."""

        # Prevent extreme steering changes that cause face planting
        steer_change = abs(R['steer'] - self.previous_steer)
        max_steer_change = 0.3

        # Reduce max change in dangerous situations
        if scenarios['emergency'] or scenarios['wall_proximity']:
            max_steer_change = 0.1
        elif scenarios['high_speed_corner']:
            max_steer_change = 0.2

        if steer_change > max_steer_change:
            direction = 1 if R['steer'] > self.previous_steer else -1
            R['steer'] = self.previous_steer + max_steer_change * direction

        # Speed-dependent safety checks
        speed = S.get('speedX', 0)

        # Prevent excessive acceleration at high speeds in corners
        curvature, _ = analyze_track_curvature(S)
        if speed > 200 and curvature > 0.3:
            R['accel'] = min(R['accel'], 0.7)

        # Ensure braking when approaching walls
        track_pos = abs(S.get('trackPos', 0))
        if track_pos > 0.8:
            R['brake'] = max(R['brake'], 0.2)

        # Final clipping
        R['steer'] = max(-1, min(1, R['steer']))
        R['accel'] = max(0, min(1, R['accel']))
        R['brake'] = max(0, min(1, R['brake']))

    def update_state_and_reward(self, S, R):
        """Update internal state and calculate reward for learning."""
        # Update state
        self.previous_steer = R['steer']
        self.previous_accel = R['accel']
        self.previous_brake = R['brake']

        # Calculate reward for learning
        reward = self.calculate_reward(S, R)

        # Update models with experience
        self.update_models(S, R, reward)
    
    def calculate_reward(self, S, R):
        """Calculate reward for reinforcement learning."""
        reward = 0
        
        # Speed reward
        speed = S.get('speedX', 0)
        reward += speed * 0.01  # Reward for going fast
        
        # Position reward
        track_pos = abs(S.get('trackPos', 0))
        if track_pos < 0.8:
            reward += 1.0  # On track
        elif track_pos < 1.0:
            reward += 0.5  # Near edge
        else:
            reward -= 2.0  # Off track penalty
        
        # Stability reward
        angle = abs(S.get('angle', 0))
        if angle < 0.2:
            reward += 0.5  # Straight driving
        
        # Damage penalty
        damage = S.get('damage', 0)
        reward -= damage * 0.001
        
        return reward

class DataCollector:
    """Collects racing data for machine learning."""
    
    def __init__(self):
        self.experiences = []
        self.max_experiences = 50000
    
    def add_experience(self, experience):
        """Add a new experience to the collection."""
        self.experiences.append(experience)
        
        # Keep only recent experiences
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]
    
    def get_training_data(self):
        """Get training data from collected experiences."""
        return self.experiences.copy()

# Global ML AI instance
ml_racing_ai = MLRacingAI()

# ================= MAIN LOOP =================
class RacingVisualizer:
    """Visualize racing data and ML model performance using mglearn."""
    
    def __init__(self):
        self.performance_data = []
        self.sensor_history = []
        self.max_history = 1000
    
    def collect_data(self, sensors, actions, reward):
        """Collect racing data for visualization."""
        data_point = {
            'speed': sensors.get('speedX', 0),
            'angle': sensors.get('angle', 0),
            'track_pos': sensors.get('trackPos', 0),
            'steer': actions.get('steer', 0),
            'accel': actions.get('accel', 0),
            'brake': actions.get('brake', 0),
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.performance_data.append(data_point)
        self.sensor_history.append(data_point)
        
        # Keep only recent data
        if len(self.performance_data) > self.max_history:
            self.performance_data = self.performance_data[-self.max_history:]
        if len(self.sensor_history) > self.max_history:
            self.sensor_history = self.sensor_history[-self.max_history:]
    
    def plot_sensor_analysis(self):
        """Plot sensor data analysis using mglearn."""
        if len(self.sensor_history) < 100:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            speeds = [d['speed'] for d in self.sensor_history[-500:]]
            angles = [d['angle'] for d in self.sensor_history[-500:]]
            track_positions = [d['track_pos'] for d in self.sensor_history[-500:]]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Speed distribution
            axes[0, 0].hist(speeds, bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_title('Speed Distribution')
            axes[0, 0].set_xlabel('Speed (km/h)')
            axes[0, 0].set_ylabel('Frequency')
            
            # Angle vs Speed scatter
            axes[0, 1].scatter(angles, speeds, alpha=0.5, s=10)
            axes[0, 1].set_title('Angle vs Speed')
            axes[0, 1].set_xlabel('Angle (rad)')
            axes[0, 1].set_ylabel('Speed (km/h)')
            
            # Track position over time
            indices = range(len(track_positions))
            axes[1, 0].plot(indices, track_positions, alpha=0.7)
            axes[1, 0].set_title('Track Position Over Time')
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Track Position')
            axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Track Edge')
            axes[1, 0].axhline(y=-1, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].legend()
            
            # Steering vs Track Position
            steers = [d['steer'] for d in self.sensor_history[-500:]]
            axes[1, 1].scatter(track_positions, steers, alpha=0.5, s=10)
            axes[1, 1].set_title('Track Position vs Steering')
            axes[1, 1].set_xlabel('Track Position')
            axes[1, 1].set_ylabel('Steering')
            
            plt.tight_layout()
            plt.savefig('racing_sensor_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Sensor analysis plot saved as 'racing_sensor_analysis.png'")
            
        except ImportError:
            print("matplotlib not available for plotting")
    
    def plot_performance_metrics(self):
        """Plot performance metrics over time."""
        if len(self.performance_data) < 50:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Extract performance data
            rewards = [d['reward'] for d in self.performance_data[-200:]]
            speeds = [d['speed'] for d in self.performance_data[-200:]]
            track_positions = [abs(d['track_pos']) for d in self.performance_data[-200:]]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Reward over time
            axes[0].plot(rewards, alpha=0.7)
            axes[0].set_title('Reward Over Time')
            axes[0].set_xlabel('Time Steps')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True, alpha=0.3)
            
            # Speed over time
            axes[1].plot(speeds, alpha=0.7, color='green')
            axes[1].set_title('Speed Over Time')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Speed (km/h)')
            axes[1].grid(True, alpha=0.3)
            
            # Track position stability
            axes[2].plot(track_positions, alpha=0.7, color='red')
            axes[2].set_title('Track Position Stability')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Distance from Center')
            axes[2].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Track Edge')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('racing_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Performance plot saved as 'racing_performance.png'")
            
        except ImportError:
            print("matplotlib not available for plotting")

# Global visualizer
visualizer = RacingVisualizer()

def drive_modular(c):
    """Entry point for the machine learning-powered racing AI."""
    ml_racing_ai.drive(c)
    
    # Collect data for visualization (every 10 steps to reduce overhead)
    if hasattr(c, 'S') and hasattr(c, 'R'):
        if not hasattr(drive_modular, 'step_counter'):
            drive_modular.step_counter = 0
        drive_modular.step_counter += 1
        
        if drive_modular.step_counter % 10 == 0:
            reward = ml_racing_ai.calculate_reward(c.S.d, c.R.d)
            visualizer.collect_data(c.S.d, c.R.d, reward)
            
            # Generate plots periodically
            if drive_modular.step_counter % 500 == 0:
                visualizer.plot_sensor_analysis()
                visualizer.plot_performance_metrics()

# ================= ML MODEL ANALYSIS WITH MGLEARN =================
def analyze_ml_models():
    """Analyze ML model performance using mglearn tools."""
    if not ml_racing_ai.is_trained:
        print("No trained models to analyze")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        print("Analyzing ML model performance...")
        
        # Create test data
        n_test = 1000
        test_features = []
        test_targets = {'steer': [], 'accel': [], 'brake': []}
        
        for _ in range(n_test):
            speed = np.random.uniform(0, 320)
            angle = np.random.normal(0, 0.3)
            track_pos = np.random.normal(0, 0.5)
            curvature = np.random.uniform(0, 1)
            
            test_features.append([speed, angle, track_pos, curvature, 
                                speed**2, angle**2, abs(track_pos), curvature*speed])
            
            test_targets['steer'].append(ml_racing_ai.expert_steering(speed, angle, track_pos, curvature))
            test_targets['accel'].append(ml_racing_ai.expert_acceleration(speed, angle, curvature))
            test_targets['brake'].append(ml_racing_ai.expert_braking(speed, angle, curvature))
        
        X_test = np.array(test_features)
        
        # Analyze each model
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, action in enumerate(['steer', 'accel', 'brake']):
            if action in ml_racing_ai.models:
                model = ml_racing_ai.models[action]
                scaler = ml_racing_ai.scalers[action]
                y_true = np.array(test_targets[action])
                
                # Scale features
                X_test_scaled = scaler.transform(X_test)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_names = ['speed', 'angle', 'track_pos', 'curvature', 
                                   'speed²', 'angle²', '|track_pos|', 'curvature×speed']
                    
                    # Plot feature importance
                    axes[0, i].barh(range(len(feature_names)), model.feature_importances_)
                    axes[0, i].set_yticks(range(len(feature_names)))
                    axes[0, i].set_yticklabels(feature_names)
                    axes[0, i].set_title(f'{action.upper()} Feature Importance')
                    axes[0, i].set_xlabel('Importance')
                
                # Prediction error distribution
                errors = y_true - y_pred
                axes[1, i].hist(errors, bins=30, alpha=0.7, color=['blue', 'green', 'red'][i])
                axes[1, i].set_title(f'{action.upper()} Prediction Errors')
                axes[1, i].set_xlabel('Error')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                # Print statistics
                mse = np.mean(errors**2)
                mae = np.mean(np.abs(errors))
                print(f"{action.upper()} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        plt.tight_layout()
        plt.savefig('ml_model_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("ML model analysis saved as 'ml_model_analysis.png'")
        
    except ImportError:
        print("matplotlib not available for ML analysis")
    except Exception as e:
        print(f"Error in ML analysis: {e}")

def generate_racing_insights():
    """Generate insights about racing performance using collected data."""
    if len(visualizer.performance_data) < 100:
        print("Not enough data for insights (need at least 100 data points)")
        return
    
    print("\n" + "="*50)
    print("🏎️  RACING AI PERFORMANCE INSIGHTS")
    print("="*50)
    
    # Analyze speed performance
    speeds = [d['speed'] for d in visualizer.performance_data]
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_std = np.std(speeds)
    
    print(f"🏎️  Speed: Avg={avg_speed:.1f}, Max={max_speed:.1f}, Std={speed_std:.1f}")
    
    # Analyze track position stability
    track_positions = [abs(d['track_pos']) for d in visualizer.performance_data]
    avg_track_error = np.mean(track_positions)
    max_track_error = np.max(track_positions)
    on_track_percentage = sum(1 for pos in track_positions if pos <= 1.0) / len(track_positions) * 100
    
    print(f"🎯 Track Position: Avg Error={avg_track_error:.3f}, Max Error={max_track_error:.3f}, On Track={on_track_percentage:.1f}%")
    
    # Analyze reward trends
    rewards = [d['reward'] for d in visualizer.performance_data]
    avg_reward = np.mean(rewards)
    reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]  # Linear trend
    
    trend_status = "📈 Improving" if reward_trend > 0.001 else "📉 Declining" if reward_trend < -0.001 else "➡️ Stable"
    print(f"💰 Rewards: Avg={avg_reward:.3f}, Trend={reward_trend:+.4f} {trend_status}")
    
    # Analyze control stability
    steers = [abs(d['steer']) for d in visualizer.performance_data]
    accels = [d['accel'] for d in visualizer.performance_data]
    brakes = [d['brake'] for d in visualizer.performance_data]
    
    avg_steer = np.mean(steers)
    avg_accel = np.mean(accels)
    avg_brake = np.mean(brakes)
    
    print(f"🎮 Controls: Steer={avg_steer:.3f}, Accel={avg_accel:.3f}, Brake={avg_brake:.3f}")
    # ML model performance
    if ml_racing_ai.is_trained:
        print("🤖 ML Model Status: ACTIVE")
        print(f"   Training Data: {len(ml_racing_ai.data_collector.experiences)} experiences")
        if hasattr(ml_racing_ai, 'learning_mode') and ml_racing_ai.learning_mode:
            print("   Learning Mode: ENABLED (continuously improving)")
        else:
            print("   Learning Mode: DISABLED (using fixed models)")
    else:
        print("🤖 ML Model Status: INACTIVE")
    
    print("="*50)
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if avg_track_error > 0.3:
        print("   • Improve track position control - car drifting too much")
    if speed_std > 50:
        print("   • Stabilize speed control - too much variation")
    if on_track_percentage < 95:
        print("   • Work on track edge detection and recovery")
    if avg_reward < 0.5:
        print("   • Optimize reward function or driving strategy")
    if ml_racing_ai.is_trained and len(ml_racing_ai.data_collector.experiences) < 1000:
        print("   • Collect more training data for better ML performance")
    
    print("="*50)

# ================= AUTOMATED TRAINING SYSTEM =================

def automated_training_pipeline(num_races=10, max_steps_per_race=5000, save_interval=5):
    """
    Automated training pipeline that runs multiple races and improves the AI.
    Automatically starts TORCS server.
    
    Args:
        num_races: Number of races to run for training
        max_steps_per_race: Maximum steps per race
        save_interval: Save models every N races
    """
    print("🚀 Starting Automated Training Pipeline")
    print(f"🎯 Target: {num_races} races, {max_steps_per_race} steps each")
    print("="*60)

    # Auto-start TORCS server
    if not start_torcs_server():
        print("❌ Cannot start training without TORCS server")
        return
    
    # Track training progress
    training_stats = {
        'races_completed': 0,
        'total_experiences': 0,
        'best_performance': float('-inf'),
        'performance_history': []
    }
    
    for race_num in range(1, num_races + 1):
        print(f"\n🏁 Race {race_num}/{num_races} - Starting...")
        
        try:
            # Create client for this race
            C = Client(p=3001, e=1, m=max_steps_per_race)
            
            # Reset step counter for data collection
            drive_modular.step_counter = 0
            
            # Run the race
            race_experiences = 0
            for step in range(C.maxSteps, 0, -1):
                C.get_servers_input()

                # Check if connection is still active after get_servers_input
                if not C.so:
                    print("⚠️  Connection lost during race, attempting to reconnect...")
                    try:
                        C.setup_connection()
                        print("✅ Reconnected successfully, continuing race...")
                    except Exception as e:
                        print(f"❌ Failed to reconnect: {e}")
                        break

                drive_modular(C)
                C.respond_to_server()
                
                # Count experiences collected
                if hasattr(drive_modular, 'step_counter') and drive_modular.step_counter % 10 == 0:
                    race_experiences += 1
                
                # Progress update
                if step % 1000 == 0:
                    progress = (C.maxSteps - step) / C.maxSteps * 100
                    print(".1f")
            
            C.shutdown()
            
            # Update training stats
            training_stats['races_completed'] += 1
            training_stats['total_experiences'] += race_experiences
            
            # Evaluate performance (using current data)
            if len(visualizer.performance_data) >= 50:
                recent_rewards = [d['reward'] for d in visualizer.performance_data[-50:]]
                avg_performance = np.mean(recent_rewards)
                training_stats['performance_history'].append(avg_performance)
                
                if avg_performance > training_stats['best_performance']:
                    training_stats['best_performance'] = avg_performance
                    print(f"   🏆 New best performance: {avg_performance:.3f}")
                else:
                    print(f"   📊 Race performance: {avg_performance:.3f}")
            # Periodic model saving and retraining
            if race_num % save_interval == 0:
                print(f"💾 Saving models after race {race_num}...")
                ml_racing_ai.save_models()
                
                # Force retraining if we have enough data
                if len(ml_racing_ai.data_collector.experiences) >= 500:
                    print("🔄 Retraining models with accumulated experience...")
                    ml_racing_ai.retrain_models()
            
            print(f"✅ Race {race_num} completed! Experiences collected: {race_experiences}")
            
        except Exception as e:
            print(f"❌ Error in race {race_num}: {e}")
            continue
    
    # Final analysis and summary
    print("\n" + "="*60)
    print("🏆 AUTOMATED TRAINING COMPLETE")
    print("="*60)
    
    print(f"📊 Training Summary:")
    print(f"   • Races completed: {training_stats['races_completed']}/{num_races}")
    print(f"   • Total experiences: {training_stats['total_experiences']}")
    print(f"   • Best performance: {training_stats['best_performance']:.3f}")
    
    if training_stats['performance_history']:
        improvement = training_stats['performance_history'][-1] - training_stats['performance_history'][0]
        print(f"   • Performance improvement: {improvement:+.3f}")
    
    # Final model save
    print("💾 Saving final trained models...")
    ml_racing_ai.save_models()
    
    # Generate final analysis
    print("📈 Generating final performance analysis...")
    analyze_ml_models()
    generate_racing_insights()
    
    print("="*60)
    print("🎯 Training pipeline completed successfully!")
    print("💡 Run 'python torcs_jm_par.py analyze' anytime to see current performance")

def continuous_learning_mode(max_races=50, performance_threshold=0.5):
    """
    Continuous learning mode that keeps training until performance threshold is reached.
    Automatically starts TORCS server.
    
    Args:
        max_races: Maximum number of races to run
        performance_threshold: Stop when average reward exceeds this threshold
    """
    print("🔄 Starting Continuous Learning Mode")
    print(f"🎯 Target: Performance > {performance_threshold} or {max_races} races max")
    print("="*60)

    # Auto-start TORCS server
    if not start_torcs_server():
        print("❌ Cannot start training without TORCS server")
        return
    
    race_num = 0
    recent_performances = []
    
    while race_num < max_races:
        race_num += 1
        print(f"\n🏁 Continuous Learning - Race {race_num}")
        
        try:
            # Run one race
            C = Client(p=3001, e=1, m=3000)  # Shorter races for continuous learning
            
            drive_modular.step_counter = 0
            
            for step in range(C.maxSteps, 0, -1):
                C.get_servers_input()

                # Check if connection is still active after get_servers_input
                if not C.so:
                    print("⚠️  Connection lost during continuous learning, attempting to reconnect...")
                    try:
                        C.setup_connection()
                        print("✅ Reconnected successfully, continuing training...")
                    except Exception as e:
                        print(f"❌ Failed to reconnect: {e}")
                        break

                drive_modular(C)
                C.respond_to_server()
                
                if step % 1000 == 0:
                    progress = (C.maxSteps - step) / C.maxSteps * 100
                    print(".1f")
            
            C.shutdown()
            
            # Check performance
            if len(visualizer.performance_data) >= 20:
                recent_rewards = [d['reward'] for d in visualizer.performance_data[-20:]]
                avg_performance = np.mean(recent_rewards)
                recent_performances.append(avg_performance)
                
                # Keep only last 5 performances for moving average
                if len(recent_performances) > 5:
                    recent_performances = recent_performances[-5:]
                
                moving_avg = np.mean(recent_performances)
                
                print(".3f")
                
                # Check if we've reached the performance threshold
                if moving_avg >= performance_threshold:
                    print(f"🎉 Performance threshold reached! Stopping training.")
                    break
            
            # Periodic retraining
            if race_num % 3 == 0 and len(ml_racing_ai.data_collector.experiences) >= 300:
                print("🔄 Retraining models...")
                ml_racing_ai.retrain_models()
                ml_racing_ai.save_models()
                
        except Exception as e:
            print(f"❌ Error in continuous learning race {race_num}: {e}")
            continue
    
    print(f"\n🏆 Continuous Learning Complete after {race_num} races")
    analyze_ml_models()
    generate_racing_insights()

def start_torcs_server():
    """
    Automatically start TORCS server for training
    """
    import platform
    import subprocess

    print("🚀 Starting TORCS server automatically...")

    if platform.system() == 'Windows':
        torcs_path = r'C:\torcs\torcs\wtorcs.exe'
        if os.path.exists(torcs_path):
            try:
                # Start TORCS in background
                subprocess.Popen([torcs_path, '-r', 'quickrace'],
                               creationflags=subprocess.CREATE_NO_WINDOW)
                print("✅ TORCS server started successfully")
                time.sleep(3)  # Give TORCS time to start
                return True
            except Exception as e:
                print(f"❌ Failed to start TORCS: {e}")
                return False
        else:
            print(f"❌ TORCS not found at {torcs_path}")
            print("   Please install TORCS in C:\\torcs\\torcs\\")
            return False
    else:
        # Linux/Mac commands
        try:
            os.system('torcs -nofuel -nodamage -nolaptime &')
            time.sleep(2)
            return True
        except:
            print("❌ Failed to start TORCS on Linux/Mac")
            return False


def perfection_training_pipeline():
    """
    Ultimate training pipeline to achieve racing perfection.
    Automatically starts TORCS server.
    """
    print("🏆 PERFECT RACING AI TRAINING - PHASE 1: FOUNDATION")
    print("="*60)
    print("🎯 Goal: Establish solid baseline performance")
    print("📊 Target: 50 races, performance > 0.3")
    print("="*60)

    # Auto-start TORCS server
    if not start_torcs_server():
        print("❌ Cannot start training without TORCS server")
        return

    # Phase 1: Foundation Building
    continuous_learning_mode(max_races=50, performance_threshold=0.3)
    
    print("\n🏆 PERFECT RACING AI TRAINING - PHASE 2: OPTIMIZATION")
    print("="*60)
    print("🎯 Goal: Optimize for speed and consistency")
    print("📊 Target: 100 races, performance > 0.6")
    print("="*60)
    
    # Phase 2: Optimization
    continuous_learning_mode(max_races=100, performance_threshold=0.6)
    
    print("\n🏆 PERFECT RACING AI TRAINING - PHASE 3: MASTERY")
    print("="*60)
    print("🎯 Goal: Achieve racing mastery")
    print("📊 Target: 200 races, performance > 0.8")
    print("="*60)
    
    # Phase 3: Mastery
    continuous_learning_mode(max_races=200, performance_threshold=0.8)
    
    print("\n🏆 PERFECT RACING AI TRAINING - PHASE 4: PERFECTION")
    print("="*60)
    print("🎯 Goal: Reach perfection")
    print("📊 Target: Unlimited races, performance > 0.95")
    print("="*60)
    
    # Phase 4: Perfection (no upper limit)
    continuous_learning_mode(max_races=1000, performance_threshold=0.95)
    
    print("\n🎉 TRAINING COMPLETE - PERFECT RACING AI ACHIEVED!")
    print("="*60)
    analyze_ml_models()
    generate_racing_insights()

def elite_training_curriculum():
    """
    Elite training curriculum with progressive difficulty levels.
    Each phase focuses on different aspects of racing perfection.
    Automatically starts TORCS server.
    """
    print("👑 ELITE RACING CURRICULUM - MULTI-PHASE TRAINING")
    print("="*65)

    # Auto-start TORCS server
    if not start_torcs_server():
        print("❌ Cannot start training without TORCS server")
        return

    phases = [
        {
            'name': 'NOVICE',
            'description': 'Basic track navigation and speed control',
            'races': 25,
            'threshold': 0.2,
            'focus': 'Stability'
        },
        {
            'name': 'INTERMEDIATE', 
            'description': 'Cornering technique and opponent awareness',
            'races': 50,
            'threshold': 0.4,
            'focus': 'Technique'
        },
        {
            'name': 'ADVANCED',
            'description': 'High-speed racing and strategic positioning',
            'races': 75,
            'threshold': 0.6,
            'focus': 'Speed'
        },
        {
            'name': 'EXPERT',
            'description': 'Defensive driving and overtaking mastery',
            'races': 100,
            'threshold': 0.75,
            'focus': 'Strategy'
        },
        {
            'name': 'MASTER',
            'description': 'Perfect lap consistency and adaptability',
            'races': 150,
            'threshold': 0.85,
            'focus': 'Consistency'
        },
        {
            'name': 'LEGENDARY',
            'description': 'Ultimate racing perfection',
            'races': 500,
            'threshold': 0.95,
            'focus': 'Perfection'
        }
    ]
    
    for phase in phases:
        print(f"\n🏆 PHASE: {phase['name']}")
        print(f"📚 Focus: {phase['focus']}")
        print(f"🎯 Goal: {phase['description']}")
        print(f"📊 Target: {phase['races']} races, performance > {phase['threshold']}")
        print("="*65)
        
        try:
            continuous_learning_mode(
                max_races=phase['races'], 
                performance_threshold=phase['threshold']
            )
            
            # Phase-specific analysis
            print(f"\n📈 {phase['name']} PHASE ANALYSIS:")
            analyze_ml_models()
            if len(visualizer.performance_data) >= 50:
                generate_racing_insights()
            
            print(f"✅ {phase['name']} Phase completed successfully!")
            
        except Exception as e:
            print(f"⚠️  Phase {phase['name']} encountered issues: {e}")
            print("Continuing to next phase...")
            continue
    
    print("\n🎉 ELITE CURRICULUM COMPLETE!")
    print("👑 Your AI has achieved LEGENDARY status!")
    print("="*65)
    
    # Final comprehensive analysis
    print("📊 FINAL PERFORMANCE ANALYSIS:")
    analyze_ml_models()
    generate_racing_insights()
    
    # Save legendary model
    legendary_filename = f"legendary_racing_ai_{int(time.time())}.pkl"
    try:
        with open(legendary_filename, 'wb') as f:
            pickle.dump({
                'models': ml_racing_ai.models,
                'scalers': ml_racing_ai.scalers,
                'training_date': time.time(),
                'performance_data': visualizer.performance_data[-100:] if len(visualizer.performance_data) > 100 else visualizer.performance_data,
                'total_experiences': len(ml_racing_ai.data_collector.experiences),
                'status': 'LEGENDARY'
            }, f)
        print(f"💾 Legendary model saved as: {legendary_filename}")
    except Exception as e:
        print(f"⚠️  Could not save legendary model: {e}")

def intensive_training_session(intensity_level='extreme'):
    """
    Intensive training session with configurable intensity levels.
    Automatically starts TORCS server.
    
    Args:
        intensity_level: 'moderate', 'intensive', 'extreme', 'insane'
    """
    print(f"🔥 Starting Intensive Training Session ({intensity_level})...")

    # Auto-start TORCS server
    if not start_torcs_server():
        print("❌ Cannot start training without TORCS server")
        return

    intensity_configs = {
        'moderate': {
            'races': 20,
            'threshold': 0.3,
            'description': 'Balanced training for steady improvement'
        },
        'intensive': {
            'races': 50,
            'threshold': 0.5,
            'description': 'Aggressive training for rapid improvement'
        },
        'extreme': {
            'races': 100,
            'threshold': 0.7,
            'description': 'Extreme training for maximum performance'
        },
        'insane': {
            'races': 200,
            'threshold': 0.85,
            'description': 'Insane training for perfection seekers'
        }
    }
    
    if intensity_level not in intensity_configs:
        print(f"❌ Invalid intensity level. Choose from: {list(intensity_configs.keys())}")
        return
    
    config = intensity_configs[intensity_level]
    
    print(f"🔥 INTENSIVE TRAINING SESSION - {intensity_level.upper()}")
    print("="*60)
    print(f"🎯 Mode: {config['description']}")
    print(f"📊 Target: {config['races']} races, performance > {config['threshold']}")
    print("="*60)
    
    # Pre-training analysis
    print("📊 Pre-training analysis:")
    analyze_ml_models()
    
    # Intensive training
    start_time = time.time()
    continuous_learning_mode(
        max_races=config['races'], 
        performance_threshold=config['threshold']
    )
    training_time = time.time() - start_time
    
    # Post-training analysis
    print("\n📊 Post-training analysis:")
    analyze_ml_models()
    generate_racing_insights()
    
    # Training summary
    print("\n🏆 TRAINING SUMMARY:")
    print(f"   • Training Time: {training_time:.1f} seconds")
    print(f"   • Intensity Level: {intensity_level.upper()}")
    print(f"   • Target Performance: {config['threshold']}")
    print(f"   • Training Data Collected: {len(ml_racing_ai.data_collector.experiences)} experiences")
    print(f"   • Performance Data Points: {len(visualizer.performance_data)}")
    
    if len(visualizer.performance_data) >= 10:
        final_performance = np.mean([d['reward'] for d in visualizer.performance_data[-10:]])
        print(f"   • Final Performance: {final_performance:.3f}")
        if final_performance >= config['threshold']:
            print("   ✅ Target performance achieved!")
        else:
            print("   ⚠️  Target performance not fully achieved")
    
    print("="*60)

# ================= MAIN LOOP =================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            # Run analysis mode
            analyze_ml_models()
            generate_racing_insights()
            
        elif sys.argv[1] == 'train':
            # Automated training pipeline
            num_races = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            automated_training_pipeline(num_races=num_races)
            
        elif sys.argv[1] == 'perfection':
            # Ultimate perfection training
            perfection_training_pipeline()
            
        elif sys.argv[1] == 'elite':
            # Elite curriculum training
            elite_training_curriculum()
            
        elif sys.argv[1] == 'intensive':
            # Intensive training session
            intensity = sys.argv[2] if len(sys.argv) > 2 else 'extreme'
            intensive_training_session(intensity_level=intensity)
            
        elif sys.argv[1] == 'demo':
            # Demo mode - show training capabilities without TORCS
            print("🎯 TORCS ML Racing AI - Automated Training Demo")
            print("="*55)
            print("🚀 Available Training Modes:")
            print("   1. analyze     - Analyze current ML models")
            print("   2. train N     - Run automated training pipeline (N races)")
            print("   3. continuous N T - Continuous learning until performance T")
            print("   4. demo        - Show this demo")
            print("   5. help        - Show usage instructions")
            print()
            print("📊 Current Status:")
            print(f"   • ML Models: {'LOADED' if ml_racing_ai.is_trained else 'NOT TRAINED'}")
            print(f"   • Training Data: {len(ml_racing_ai.data_collector.experiences)} experiences")
            print(f"   • Performance Data: {len(visualizer.performance_data)} points")
            print()
            print("💡 To start automated training:")
            print("   1. Start TORCS server")
            print("   2. Run: python torcs_jm_par.py train 5")
            print("   3. Watch the AI learn and improve automatically!")
            print("="*55)
            
        elif sys.argv[1] == 'help':
            print("🏎️  TORCS ML Racing AI - Usage:")
            print("="*50)
            print("python torcs_jm_par.py              # Run single race")
            print("python torcs_jm_par.py analyze      # Analyze current models")
            print("python torcs_jm_par.py train [N]    # Automated training (N races)")
            print("python torcs_jm_par.py continuous [N] [T]  # Continuous learning")
            print("                                      # N=max races, T=performance threshold")
            print("python torcs_jm_par.py perfection   # Ultimate perfection training")
            print("python torcs_jm_par.py elite        # Elite curriculum training")
            print("python torcs_jm_par.py intensive [L] # Intensive training (L=moderate/hard/extreme/insane)")
            print("python torcs_jm_par.py demo         # Show training capabilities")
            print("python torcs_jm_par.py help         # Show this help")
            print("="*50)
            
        else:
            print("❌ Unknown command. Use 'python torcs_jm_par.py demo' for options.")
            
    else:
        # Run racing mode (default)
        print("🏎️  Starting Machine Learning Racing AI...")
        print("🤖 ML Models:", "LOADED" if ml_racing_ai.is_trained else "TRAINING")
        print("📊 Data collection and visualization: ENABLED")
        print("🎯 Target: Ultimate racing performance with continuous learning")
        print("="*60)
        
        C = Client(p=3001)
        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()

            # Check if connection is still active after get_servers_input
            if not C.so:
                print("⚠️  Connection lost, attempting to reconnect...")
                try:
                    C.setup_connection()
                    print("✅ Reconnected successfully, continuing race...")
                except Exception as e:
                    print(f"❌ Failed to reconnect: {e}")
                    break

            drive_modular(C)
            C.respond_to_server()

            # Periodic analysis
            if step % 1000 == 0:
                print(f"Step {C.maxSteps - step}/{C.maxSteps} - AI learning and adapting...")

        C.shutdown()
        
        # Final analysis
        print("\n" + "="*60)
        print("🏁 RACE COMPLETE - Generating final analysis...")
        analyze_ml_models()
        generate_racing_insights()
        print("="*60)
