"""
TORCS Client Module

Handles communication with the TORCS server, including socket management,
data parsing, and driver actions.
"""

import socket
import time
from typing import Optional, Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

PI = 3.141592653589793
DATA_SIZE = 2**17

class ServerState:
    """Represents the current state reported by the TORCS server."""

    def __init__(self) -> None:
        self.servstr: str = ""
        self.d: Dict[str, Any] = {}

    def parse_server_str(self, server_string: str) -> None:
        """Parse one complete SCR packet without retaining stale fields."""
        text = server_string.strip().strip('\x00')
        if not text.startswith('(') or not text.endswith(')'):
            raise ValueError("SCR packet must be a parenthesized sensor message")
        fields: Dict[str, Any] = {}
        for item in text[1:-1].split(')('):
            words = item.split()
            if len(words) < 2 or not words[0]:
                raise ValueError("SCR packet contains an invalid sensor field")
            fields[words[0]] = destringify(words[1:])
        if not fields:
            raise ValueError("SCR packet contains no sensor fields")
        self.servstr = text
        self.d = fields

    def __repr__(self) -> str:
        return self.fancyout()

    def fancyout(self) -> str:
        """Specialty output for ServerState monitoring."""
        out = ""
        sensors = [
            'stucktimer', 'fuel', 'distRaced', 'distFromStart', 'opponents',
            'wheelSpinVel', 'z', 'speedZ', 'speedY', 'speedX', 'targetSpeed',
            'rpm', 'skid', 'slip', 'track', 'trackPos', 'angle'
        ]

        for k in sensors:
            if k not in self.d:
                continue
            if isinstance(self.d.get(k), list):
                if k == 'track':
                    raw_tsens = ['%.1f' % x for x in self.d['track']]
                    strout = ' '.join(raw_tsens[:9]) + '_' + raw_tsens[9] + '_' + ' '.join(raw_tsens[10:])
                elif k == 'opponents':
                    strout = ''
                    for osensor in self.d['opponents']:
                        if osensor > 190: oc = '_'
                        elif osensor > 90: oc = '.'
                        elif osensor > 39: oc = chr(int(osensor/2) + 97 - 19)
                        elif osensor > 13: oc = chr(int(osensor) + 65 - 13)
                        elif osensor > 3: oc = chr(int(osensor) + 48 - 3)
                        else: oc = '?'
                        strout += oc
                    strout = ' -> ' + strout[:18] + ' ' + strout[18:] + ' <-'
                else:
                    strlist = [str(i) for i in self.d[k]]
                    strout = ', '.join(strlist)
            else:
                strout = self._format_sensor(k)
            out += f"{k}: {strout}\n"
        return out

    def _format_sensor(self, k: str) -> str:
        """Format individual sensor values."""
        value = self.d[k]
        if k == 'gear':
            gs = '_._._._._._._._._'
            p = int(value) * 2 + 2
            l = '%d' % value
            if l == '-1': l = 'R'
            if l == '0': l = 'N'
            return gs[:p] + '(%s)' % l + gs[p+3:]
        elif k == 'damage':
            return '%6.0f %s' % (value, bargraph(value, 0, 10000, 50, '~'))
        elif k == 'fuel':
            return '%6.0f %s' % (value, bargraph(value, 0, 100, 50, 'f'))
        elif k == 'speedX':
            cx = 'X' if value >= 0 else 'R'
            return '%6.1f %s' % (value, bargraph(value, -30, 300, 50, cx))
        elif k == 'speedY':
            return '%6.1f %s' % (value, bargraph(value * -1, -25, 25, 50, 'Y'))
        elif k == 'speedZ':
            return '%6.1f %s' % (value, bargraph(value, -13, 13, 50, 'Z'))
        elif k == 'z':
            return '%6.3f %s' % (value, bargraph(value, .3, .5, 50, 'z'))
        elif k == 'trackPos':
            cx = '<' if value >= 0 else '>'
            return '%6.3f %s' % (value, bargraph(value * -1, -1, 1, 50, cx))
        elif k == 'stucktimer':
            if value:
                return '%3d %s' % (value, bargraph(value, 0, 300, 50, "'"))
            else:
                return 'Not stuck!'
        elif k == 'rpm':
            g = self.d.get('gear', 0)
            if g < 0:
                g = 'R'
            else:
                g = '%1d' % g
            return bargraph(value, 0, 10000, 50, g)
        elif k == 'angle':
            asyms = [
                r"  !  ", r".|'  ", r"./'  ", r"_.-  ", r".--  ", r"..-  ",
                r"---  ", r".__  ", r"-._  ", r"'-.  ", r"'\.  ", r"'|.  ",
                r"  |  ", r"  .|'", r"  ./'", r"  .-'", r"  _.-", r"  __.",
                r"  ---", r"  --.", r"  -._", r"  -..", r"  '\.", r"  '|."  ]
            rad = value
            deg = int(rad * 180 / PI)
            symno = int(.5 + (rad + PI) / (PI / 12))
            symno = symno % (len(asyms) - 1)
            return '%5.2f %3d (%s)' % (rad, deg, asyms[symno])
        elif k == 'skid':
            frontwheelradpersec = self.d.get('wheelSpinVel', [0])[0]
            skid = 0
            if frontwheelradpersec:
                skid = .5555555555 * self.d.get('speedX', 0) / frontwheelradpersec - .66124
            return bargraph(skid, -.05, .4, 50, '*')
        elif k == 'slip':
            wheelSpinVel = self.d.get('wheelSpinVel', [0, 0, 0, 0])
            frontwheelradpersec = wheelSpinVel[0]
            slip = 0
            if frontwheelradpersec:
                slip = ((wheelSpinVel[2] + wheelSpinVel[3]) -
                       (wheelSpinVel[0] + wheelSpinVel[1]))
            return bargraph(slip, -5, 150, 50, '@')
        else:
            return str(value)


class DriverAction:
    """Represents the driver's intended actions to send to the server."""

    def __init__(self) -> None:
        self.actionstr: str = ""
        self.d: Dict[str, Any] = {
            'accel': 0.8,
            'brake': 0,
            'clutch': 0,
            'gear': 3,
            'steer': 0.1,
            'focus': [-90, -45, 0, 45, 90],
            'meta': 0
        }

    def clip_to_limits(self) -> None:
        """Clip values to sensible limits."""
        self.d['steer'] = max(-1, min(1, self.d['steer']))
        self.d['brake'] = max(0, min(1, self.d['brake']))
        self.d['accel'] = max(0, min(1, self.d['accel']))
        self.d['clutch'] = max(0, min(1, self.d['clutch']))
        if self.d['accel'] > 0 and self.d['brake'] > 0:
            if self.d['brake'] >= self.d['accel']:
                self.d['accel'] = 0.0
            else:
                self.d['brake'] = 0.0
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if not isinstance(self.d['focus'], list) or min(self.d['focus']) < -180 or max(self.d['focus']) > 180:
            self.d['focus'] = 0

    def __repr__(self) -> str:
        self.clip_to_limits()
        out = ""
        for k, v in self.d.items():
            out += f'({k} '
            if not isinstance(v, list):
                out += '%.3f' % v
            else:
                out += ' '.join(str(x) for x in v)
            out += ')'
        return out

    def fancyout(self) -> str:
        """Specialty output for monitoring driver's actions."""
        out = ""
        od = self.d.copy()
        od.pop('gear', None)
        od.pop('meta', None)
        od.pop('focus', None)
        for k in sorted(od):
            if k in ['clutch', 'brake', 'accel']:
                strout = '%6.3f %s' % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == 'steer':
                strout = '%6.3f %s' % (od[k], bargraph(od[k] * -1, -1, 1, 50, 'S'))
            else:
                strout = str(od[k])
            out += f"{k}: {strout}\n"
        return out


class Client:
    """TORCS client for communicating with the racing server."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 sid: Optional[str] = None, max_episodes: Optional[int] = None,
                 trackname: Optional[str] = None, stage: Optional[int] = None,
                 debug: Optional[bool] = None, max_steps: Optional[int] = None,
                 vision: bool = False, connect: bool = True,
                 connect_attempts: int = 5, connect_timeout: float = 1.0) -> None:
        self.vision = vision
        self.host = host or 'localhost'
        self.port = port or 3001
        self.sid = sid or 'SCR'
        self.maxEpisodes = max_episodes or 1
        self.trackname = trackname or 'unknown'
        self.stage = stage if stage is not None else 3
        self.debug = debug or False
        self.maxSteps = max_steps or 100000
        if connect_attempts < 1 or connect_timeout <= 0:
            raise ValueError("connect_attempts must be positive and connect_timeout must be > 0")
        self.connect_attempts = connect_attempts
        self.connect_timeout = connect_timeout
        self.closed_reason: Optional[str] = None
        self.S = ServerState()
        self.R = DriverAction()
        self.so: Optional[socket.socket] = None
        if connect:
            self.setup_connection()

    def setup_connection(self) -> None:
        """Establish connection to TORCS server."""
        self.shutdown()
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(self.connect_timeout)
        except socket.error as e:
            raise ConnectionError(f"Could not create socket: {e}") from e

        allowed_hosts = {self.host}
        try:
            allowed_hosts.add(socket.gethostbyname(self.host))
        except socket.gaierror:
            pass
        for attempt in range(1, self.connect_attempts + 1):
            angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = f'{self.sid}(init {angles})'

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as e:
                self.shutdown()
                raise ConnectionError(f"Send error: {e}") from e

            try:
                sockdata, addr = self.so.recvfrom(DATA_SIZE)
                sockdata = sockdata.decode('utf-8')
            except (socket.timeout, ConnectionResetError):
                logger.info(
                    "Waiting for server on %s (attempt %s/%s)",
                    self.port, attempt, self.connect_attempts,
                )
                time.sleep(min(0.25, self.connect_timeout / 2))
                continue
            except (socket.error, UnicodeDecodeError) as e:
                self.shutdown()
                raise ConnectionError(f"Error receiving server identification: {e}") from e

            if addr[0] not in allowed_hosts:
                logger.warning("Ignoring SCR packet from unexpected peer %s", addr[0])
                continue
            if '***identified***' in sockdata:
                logger.info(f"Client connected on {self.port}")
                return
        self.shutdown()
        raise TimeoutError(
            f"TORCS did not identify the client on {self.host}:{self.port} "
            f"after {self.connect_attempts} attempts"
        )

    def get_servers_input(self) -> None:
        """Receive and parse input from server."""
        if not self.so:
            return

        while True:
            try:
                sockdata, addr = self.so.recvfrom(DATA_SIZE)
                sockdata = sockdata.decode('utf-8')
            except socket.timeout as exc:
                raise TimeoutError(f"Timed out waiting for TORCS on port {self.port}") from exc
            except (socket.error, UnicodeDecodeError) as exc:
                raise ConnectionError(f"Error receiving from TORCS: {exc}") from exc

            try:
                expected_hosts = {self.host, socket.gethostbyname(self.host)}
            except socket.gaierror:
                expected_hosts = {self.host}
            if addr[0] not in expected_hosts:
                logger.warning("Ignoring SCR packet from unexpected peer %s", addr[0])
                continue

            if '***identified***' in sockdata:
                logger.info(f"Client connected on {self.port}")
                continue
            elif '***shutdown***' in sockdata:
                logger.info(f"Server stopped race on {self.port}. Position: {self.S.d.get('racePos', 'unknown')}")
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                logger.info(f"Server restarted race on {self.port}")
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                try:
                    self.S.parse_server_str(sockdata)
                except ValueError as exc:
                    raise ValueError(f"Malformed SCR telemetry: {exc}") from exc
                if self.debug:
                    print(self.S)
                break

    def respond_to_server(self) -> None:
        """Send driver actions to server."""
        if not self.so:
            return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as e:
            raise ConnectionError(f"Error sending to server: {e}") from e
        if self.debug:
            print(self.R.fancyout())

    def shutdown(self) -> None:
        """Close connection and shutdown."""
        if self.so:
            logger.info(f"Shutting down connection on port {self.port}")
            self.so.close()
            self.so = None


def destringify(s: Any) -> Any:
    """Convert string to appropriate type."""
    if not s:
        return s
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    elif isinstance(s, list):
        if len(s) == 0:
            return s
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]
    return s


def bargraph(x: float, mn: float, mx: float, w: int, c: str = 'X') -> str:
    """Draw ASCII bar graph."""
    if not w:
        return ''
    x = max(mn, min(mx, x))
    tx = mx - mn
    if tx <= 0:
        return 'backwards'
    upw = tx / float(w)
    if upw <= 0:
        return 'what?'
    negpu = -x + min(0, mx) if mn < 0 and x < 0 else 0
    pospu = x - max(0, mn) if mx > 0 and x > 0 else 0
    negnonpu = -mn + min(0, mx) if mn < 0 else 0
    posnonpu = mx - max(0, mn) if mx > 0 else 0
    nnc = int(negnonpu / upw) * '-'
    npc = int(negpu / upw) * c
    ppc = int(pospu / upw) * c
    pnc = int(posnonpu / upw) * '_'
    return f'[{nnc}{npc}{ppc}{pnc}]'


def clip(v: float, lo: float, hi: float) -> float:
    """Clip value to range."""
    return max(lo, min(hi, v))
