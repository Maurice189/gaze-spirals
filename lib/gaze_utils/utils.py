def add_frame_col(gaze, fps, remove_offset=False):
    if 'FRAME' in gaze:
        raise ValueError('FRAME column already exists!')
    if remove_offset:
        offset = gaze['TIME_MS'].values[0]
    else:
        offset = 0
    gaze['FRAME'] = gaze['TIME_MS'].apply(lambda t_ms: (t_ms-offset)*fps/1000).round()
    gaze['FRAME'] = gaze['FRAME'] + 1
    return gaze
