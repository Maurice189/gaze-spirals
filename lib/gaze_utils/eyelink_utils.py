import pandas as pd


def parse_eyelink_fixations(fixation_events, separator='\t', check_consistency=True):
    efix_events = [s for s in fixation_events if s.startswith('EFIX')]
    if check_consistency:
        sfix_events = [s for s in fixation_events if s.startswith('SFIX')]
        assert len(sfix_events) == len(efix_events), "Number of SFIX events does not match number EFIX events!"

    efix_data = map(lambda s: s.split(separator)[1:], efix_events)
    columns = ['EYE', 'START_TIME', 'END_TIME', 'DURATION', 'AVG_X', 'AVG_Y', 'AVG_PUPIL']
    df = pd.DataFrame(data=efix_data, columns=columns)
    # Convert every column except for 'EYE' to numeric
    df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric)
    return df


def parse_eyelink_saccades(saccade_events, separator='\t', check_consistency=True):
    esacc_events = [s for s in saccade_events if s.startswith('ESACC')]
    if check_consistency:
        ssacc_events = [s for s in saccade_events if s.startswith('SSACC')]
        assert len(ssacc_events) == len(esacc_events), "Number of SSAC events does not match number ESACC events!"

    esacc_data = map(lambda s: s.split(separator)[1:], esacc_events)
    columns = ['EYE', 'START_TIME', 'END_TIME', 'DURATION', 'START_X', 'START_Y', 'END_X', 'END_Y', 'AMP', 'PEAK_VEL']
    df = pd.DataFrame(data=esacc_data, columns=columns)
    # Convert every column except for 'EYE' to numeric
    df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric)
    return df


def parse_eyelink_blinks(blink_events, separator='\t', check_consistency=True):
    eblink_events = [s for s in blink_events if s.startswith('EBLINK')]
    if check_consistency:
        sblink_events = [s for s in blink_events if s.startswith('SBLINK')]
        assert len(sblink_events) == len(eblink_events), "Number of SBLINK events does not match number EBLINK events!"

    eblink_data = map(lambda s: s.split(separator)[1:], eblink_events)
    columns = ['EYE', 'START_TIME', 'END_TIME', 'DURATION']
    df = pd.DataFrame(data=eblink_data, columns=columns)
    # Convert every column except for 'EYE' to numeric
    df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric)
    return df

