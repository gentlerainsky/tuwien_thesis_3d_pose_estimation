import datetime


class Timer:
    def __init__(self):
        self.start_time = None
        self.finish_time = None
        self.start_lap = None

    def start(self):
        self.start_time = datetime.datetime.now()
        self.finish_time = self.start_time
    
    def finish(self):
        self.finish_time = datetime.datetime.now()

    def lap(self):
        self.start_lap = self.finish_time
        self.finish_time = datetime.datetime.now()

    def __str__(self):
        time_used = self.finish_time - self.start_time
        time_mins = int(time_used.total_seconds() // 60)
        time_hours = int(time_mins // 60)
        time_mins = int(time_mins % 60)
        time_sec = int(time_used.total_seconds() % 60)
        out = f'Total Time: {time_hours:02d}:{time_mins:02d}:{time_sec:02d} Hours\n'
        lap_time_used = self.finish_time - self.start_lap
        lap_time_mins = int(lap_time_used.total_seconds() // 60)
        lap_time_sec = int(lap_time_used.total_seconds() % 60)
        out += f'Lap Time: {lap_time_mins:02d}:{lap_time_sec:02d} Mins'
        return out
