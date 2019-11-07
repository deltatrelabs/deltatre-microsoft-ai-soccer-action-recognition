# This is the script for cutting out clips with shots, no-shots and attacks and putting them into data_dir

from config import *

import mPyPl as mp
from mpyplx import *
import os
from pipe import Pipe
from moviepy.editor import *
import itertools
import numpy as np
import json
import argparse
from distutils import util

def convtime(s):
    t = s.split(':') | mp.select(float) | mp.as_list
    return t[2]+t[1]*60+t[0]*60*60

parser = argparse.ArgumentParser()
parser.add_argument('--debugMatchId', type=int, help='Produce clips for the specified match only', required=False)
parser.add_argument('--debugMaxClips', type=int, help='Produce up to N clips for the specified match/half time only', required=False)
parser.add_argument('--debug', help='Enable detailed traces', required=False, default='False')
parser.add_argument('--debugShotsOnly', help='Produce Shots clip only', required=False, default='False')
parser.add_argument('--goalClipEnabled', help='Toggle "Goal" Shot clips', required=False, default='True')
parser.add_argument('--overwriteExisting', help='If a clip exists, overwrite or not', required=False, default='False')
parser.add_argument('--clipTrimTime', help='Specify half-clip duration', required=False, default=2.5)
parser.add_argument('--clipTrimAttackTimeRange', help='Specify +/- X seconds centered on event time', required=False, default=10)
parser.add_argument('--clipTrimAttackStart', help='Specify when attack clip starts (before event time) in seconds', required=False, default=1)
parser.add_argument('--clipTrimAttackEnd', help='Specify when attack clip ends (after event time) in seconds', required=False, default=4)
parser.add_argument('--noShotClipDuration', help='Specify no-shot clip duration in seconds', required=False, default=5)
parser.add_argument('--noShotClipInterval', help='Specify how many no-shot to take between 2 shot events', required=False, default=10)
parser.add_argument('--resizeClips', help='Toggle clip resizing', required=False, default='False')
args = parser.parse_args()

shotFolder = "shot"
noShotFolder = "noshot"
attacksFolder = "attack"

debug = util.strtobool(args.debug)
shotsOnly = util.strtobool(args.debugShotsOnly)
overwriteExist = util.strtobool(args.overwriteExisting)
goalClipEnabled = util.strtobool(args.goalClipEnabled)
resizeClips = util.strtobool(args.resizeClips)

shotClips = []
noshotClips = []
attackClips = []

print(f'         Debug: {debug}')
print(f'     ShotsOnly: {shotsOnly}')
print(f'   GoalEnabled: {goalClipEnabled}')
print(f'     Overwrite: {overwriteExist}')
print(f'Clip Trim-time: {args.clipTrimTime}')

def shotFilter(tags):
    if goalClipEnabled:
        return "shot" in tags or "shot_goal" in tags
    else:
        return "shot" in tags

def attackFilter(tags):
    return "attack" in tags

def addClipToList(list, matchId, n, h, eventMatchPhase, x, mt, dt, satoff, startClip, endClip, isGoal, goalOff, shotTypes, prev_time, next_time):
    clipMeta = {"matchId": matchId,
                "n": n,
                "Half": h,
                "EventMatchPhase": eventMatchPhase,
                "MatchTime": x,
                "PhaseStartTime": mt,
                "Kickoff": dt,
                "SatOff": satoff,
                "GoalOff": goalOff,
                "ClipStart": startClip,
                "ClipEnd": endClip,
                "IsGoal": isGoal,
                "Type": shotTypes,
                "Previous_time": prev_time,
                "Next_time": next_time
        }
    list.append(clipMeta)

# Convert back seconds to HH:mm:ss, for easily jump in marks.jsonl
def GetTime(seconds):
    hh = seconds//(60*60)
    mm = (seconds-hh*60*60)//60
    ss = seconds-(hh*60*60)-(mm*60)
    return (f"{int(hh):0>2d}:{int(mm):0>2d}:{int(ss):0>2d}")


matches = from_json(os.path.join(source_dir,'matches.json'))

for match in matches:

    if args.debugMatchId != None:
        if match["Id"] != args.debugMatchId:
            #print(f"Skipping match {match['Id']} - Debug mode for match {args.debugMatchId}")
            continue
        else:
            print(f"\nDebug mode for match '{args.debugMatchId}'")

    dt = convtime(match["ClippingSettings"]["Kickoff"])
    satoff = match["ClippingSettings"]["SatOffset"]
    SecondHalfDelay = match["ClippingSettings"]["HalfDelay"]
    matchDay = match["MatchDay"]
    goaloff = match["ClippingSettings"]["ActionOverrides"]["Goal"]["ActionDelay"]

    competition = match["Competition"].lower()
    season = match["Season"].replace("/", "")

    h = max(0, int(match["Half"]))

    print(f"Processing {match['Video']} ({match['Id']})\n")
    videoFilePath = os.path.join(source_dir,match["Video"])
    marksFilePath = os.path.join(source_dir,"Marks.jsonl")
    video = VideoFileClip(videoFilePath)

    data = []
    with open(marksFilePath,'r') as f:
        for cnt, line in enumerate(f):
            data.append(json.loads(line))

    # Get list of times when match halves start
    mt = data | mp.filter('eventType', lambda x: "start" in x) | mp.select_field('matchTime') | mp.as_list
    mt = convtime(mt[h])

    # Shots are fine, added also Goal which are not marked as shots
    cuts = data | mp.filter('eventType', shotFilter) | mp.apply('matchTime', 'start', convtime)

    # Consider, for negative examples, also Attacks, which are Shots nearby the goal area (or in other places of the field)
    # They are also filtered (later), removing those overlapping with shots/goals
    attacks = data | mp.filter('eventType', attackFilter) | mp.apply('matchTime', 'start', convtime)

    clipTrimTime = float(args.clipTrimTime)                       # +/- X seconds centered on event time

    clipTrimAttackTimeRange = float(args.clipTrimAttackTimeRange) # +/- X seconds centered on event time
    clipTrimAttackStart = float(args.clipTrimAttackStart)         # when attack clip starts (before event time)
    clipTrimAttackEnd = float(args.clipTrimAttackEnd)             # when attack clip ends (after event time)

    noShotClipDuration = float(args.noShotClipDuration)
    noShotClipInterval = float(args.noShotClipInterval)

    n=0
    prev_time = mt
    for cutInfo in cuts:
        
        if args.debugMaxClips != None and n >= args.debugMaxClips:
            print(f"Debug mode - Reached max number of clips ({args.debugMaxClips})\n")
            break

        x = cutInfo['start']
              
        eventMatchPhase = cutInfo["phase"]

        # write out correct "shot" clip

        isGoal = 'shot_goal' in cutInfo['eventType']
        shotType = cutInfo['eventType']

        additionalShiftForGoal = 0
        shotKindForFileName = "Shot"
        if isGoal:
            shotKindForFileName = "Goal"
            additionalShiftForGoal = goaloff

        startClip = x + dt - mt - satoff - clipTrimTime - additionalShiftForGoal
        endClip =  x + dt - mt - satoff + clipTrimTime - additionalShiftForGoal

        shotTypeFolder = shotType
        shotTypesForFileName = shotType

        # Skip cuts that are in the wrong "half" of the video
        if eventMatchPhase != h:
            continue

        if eventMatchPhase==0 and h == eventMatchPhase and startClip > video.duration:
            print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
            break

        if eventMatchPhase==1 and h == eventMatchPhase and startClip <= 0:
            print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
            continue

        if debug:
            print("Shot\n====================================================")
            print(f"n:               {n}\n" +
                  f"Half:            {h+1}\n" +
                  f"EventMatchPhase: {eventMatchPhase+1}\n" +
                  f"MatchTime:       {GetTime(x)}\n" +
                  f"PhaseStartTime:  {GetTime(mt)}\n" +
                  f"Kickoff:         {GetTime(dt)}\n" +
                  f"SatOff:          {GetTime(satoff)}\n" +
                  f"ClipStart:       {GetTime(startClip)}\n" +
                  f"ClipEnd:         {GetTime(endClip)}\n" +
                  f"IsGoal:          {isGoal}\n" +
                  f"Type:            {shotTypesForFileName}\n")

        if startClip < 0:
            print(f"StartClip < 0. Forcing to 0.0\n")
            startClip = 0

        if endClip > video.duration:
            print(f"EndClip > video.duration. Forcing to video.duration\n")
            endClip = video.duration

        addClipToList(shotClips, match["Id"], n, h+1, eventMatchPhase+1, GetTime(x), GetTime(mt), GetTime(dt), GetTime(satoff), GetTime(startClip), GetTime(endClip), isGoal, goaloff, shotType, "", "")

        cut = video.subclip(startClip, endClip)
        outputDir = os.path.join(base_dir, "processed", shotFolder, shotTypeFolder)
        os.makedirs(outputDir, exist_ok=True)
        videoOutFileName = os.path.join(outputDir,f"{match['Id']}_{h+1}_{n}_{shotTypesForFileName}.full.mp4")
        if not os.path.isfile(videoOutFileName) or overwriteExist == True: # skip if exists
            cut.write_videofile(videoOutFileName)

        if resizeClips:
            resizedOutputFile = videoOutFileName.replace(".full.mp4",".resized.mp4")
            resized = cut.fx(vfx.resize, width=video_size[0])
            if not os.path.isfile(resizedOutputFile) or overwriteExist == True: # skip if exists
                resized.write_videofile(resizedOutputFile)

        # Skips No-Shots in DEBUG
        if shotsOnly == False:

            # Write out "noshot" clip (in between this and last shot)

            nx = (prev_time+x)/2
            if x - nx > noShotClipInterval:
               
                startClip = nx + dt - mt + satoff
                endClip = nx + dt - mt + satoff + noShotClipDuration

                # Skip cuts that are in the wrong "half" of the video
                if eventMatchPhase != h:
                    continue

                if eventMatchPhase==0 and h == eventMatchPhase and startClip > video.duration:
                    #print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                    break
                if eventMatchPhase==1 and h == eventMatchPhase and startClip <= 0:
                    #print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                    continue

                if debug:
                    print("No-shot\n====================================================")
                    print(f"n:               {n}\n" +
                          f"Half:            {h+1}\n" +
                          f"EventMatchPhase: {eventMatchPhase+1}\n" +
                          f"Prev:            {GetTime(prev_time)}\n" +
                          f"Next:            {GetTime(nx)}\n" +
                          f"PhaseStartTime:  {GetTime(mt)}\n" +
                          f"Kickoff:         {GetTime(dt)}\n" +
                          f"SatOff:          {GetTime(satoff)}\n" +
                          f"ClipStart:       {GetTime(startClip)}\n" +
                          f"ClipEnd:         {GetTime(endClip)}\n")

                if startClip < 0:
                    print(f"StartClip < 0. Forcing to 0.0\n")
                    startClip = 0

                if endClip > video.duration:
                    print(f"EndClip > video.duration. Forcing to video.duration\n")
                    endClip = video.duration

                addClipToList(noshotClips, match["Id"], n, h+1, eventMatchPhase+1, GetTime(x), GetTime(mt), GetTime(dt), GetTime(satoff), GetTime(startClip), GetTime(endClip), "false", goaloff, "", GetTime(prev_time), GetTime(nx))

                cut = video.subclip(startClip, endClip)

                outputDir = os.path.join(base_dir, "processed", noShotFolder)
                os.makedirs(outputDir, exist_ok=True)
                videoOutFileName = os.path.join(outputDir, f"{match['Id']}_{h+1}_{n}.full.mp4")
                if not os.path.isfile(videoOutFileName) or overwriteExist == True: # skip if exists
                    cut.write_videofile(videoOutFileName)

                if resizeClips:
                    resizedOutputFile = videoOutFileName.replace(".full.mp4",".resized.mp4")
                    resized = cut.fx(vfx.resize, width=video_size[0])
                    if not os.path.isfile(resizedOutputFile) or overwriteExist == True: # skip if exists
                        resized.write_videofile(resizedOutputFile)
            else:
                print("WARNING: Time between shots too small, cannot cut no-shot\n")

        n+=1
        prev_time = x

    # Dump lists
    if len(shotClips)>0:
        with open(os.path.join(base_dir, "processed", shotFolder, 'shots.txt'), 'w') as outfile:  
            json.dump(shotClips, outfile, indent=4)

    if len(noshotClips)>0:
        with open(os.path.join(base_dir, "processed", noShotFolder, 'noshots.txt'), 'w') as outfile:  
            json.dump(noshotClips, outfile, indent=4)

    # Skips Attacks Shots in DEBUG
    if shotsOnly == True:
        video.reader.close()
        video.audio.reader.close_proc()
        continue

    print("Attacks\n====================================================")

    n=0
    cutList = list(cuts)
    for attackInfo in attacks:

        if args.debugMaxClips != None and n >= args.debugMaxClips:
            print(f"Debug mode - Reached max number of clips ({args.debugMaxClips})\n")
            break

        x = attackInfo['start']

        doClip = True
        for cutInfo in cutList:
            attackClipOverlapStart = cutInfo['start'] - clipTrimAttackTimeRange
            attackClipOverlapEnd = cutInfo['start'] + clipTrimAttackTimeRange
            if x >= attackClipOverlapStart and x <= attackClipOverlapEnd:
                print(f'attackEvent: {x} shotEvent: {cutInfo["start"]} shotClipStart: {attackClipOverlapStart} shotClipEnd: {attackClipOverlapEnd}')
                print(f"WARNING: Attack {attackInfo['MarkGuid']} overlaps with Shot {cutInfo['MarkGuid']}")
                doClip = False
                break

        if doClip:

            startClip = x + dt - mt - satoff - clipTrimAttackStart
            endClip =  x + dt - mt - satoff + clipTrimAttackEnd

            # Skip cuts that are in the wrong "half" of the video
            if eventMatchPhase != h:
                continue

            if eventMatchPhase==0 and h == eventMatchPhase and startClip > video.duration:
                print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                break
            if eventMatchPhase==1 and h == eventMatchPhase and startClip < 0:
                #print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                continue

            if debug:
                print(f"n:               {n}\n" +
                    f"Half:            {h+1}\n" +
                    f"EventMatchPhase: {eventMatchPhase+1}\n" +
                    f"MatchTime:       {GetTime(x)}\n" +
                    f"PhaseStartTime:  {GetTime(mt)}\n" +
                    f"Kickoff:         {GetTime(dt)}\n" +
                    f"SatOff:          {GetTime(satoff)}\n" +
                    f"ClipStart:       {GetTime(startClip)}\n" +
                    f"ClipEnd:         {GetTime(endClip)}\n")

            startClip = x + dt - mt - satoff - clipTrimAttackStart
            endClip =  x + dt - mt - satoff + clipTrimAttackEnd

            # Skip cuts that are in the wrong "half" of the video
            if eventMatchPhase != h:
                continue

            if eventMatchPhase==0 and h == eventMatchPhase and startClip > video.duration:
                print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                break
            if eventMatchPhase==1 and h == eventMatchPhase and startClip < 0:
                #print(f"Cut in the wrong 'half' of the video (h={h+1}). Skipping\n")
                continue

            if debug:
                print(f"n:               {n}\n" +
                    f"Half:            {h+1}\n" +
                    f"EventMatchPhase: {eventMatchPhase+1}\n" +
                    f"MatchTime:       {GetTime(x)}\n" +
                    f"PhaseStartTime:  {GetTime(mt)}\n" +
                    f"Kickoff:         {GetTime(dt)}\n" +
                    f"SatOff:          {GetTime(satoff)}\n" +
                    f"ClipStart:       {GetTime(startClip)}\n" +
                    f"ClipEnd:         {GetTime(endClip)}\n")

            if startClip < 0:
                print(f"StartClip < 0. Forcing to 0.0\n")
                startClip = 0

            if endClip > video.duration:
                print(f"EndClip > video.duration. Forcing to video.duration\n")
                endClip = video.duration

            addClipToList(attackClips, match["Id"], n, h+1, eventMatchPhase+1, GetTime(x), GetTime(mt), GetTime(dt), GetTime(satoff), GetTime(startClip), GetTime(endClip), "false", goaloff, "", "", "")

            cut = video.subclip(startClip, endClip)
            outputDir = os.path.join(base_dir, "processed", attacksFolder)
            os.makedirs(outputDir, exist_ok=True)
            videoOutFileName = os.path.join(outputDir, f"{match['Id']}_{h+1}_{n}.full.mp4")
            if not os.path.isfile(videoOutFileName) or overwriteExist == True: # skip if exists
                cut.write_videofile(videoOutFileName)

            if resizeClips:
                resizedOutputFile = videoOutFileName.replace(".full.mp4",".resized.mp4")
                resized = cut.fx(vfx.resize, width=video_size[0])
                if not os.path.isfile(resizedOutputFile) or overwriteExist == True: # skip if exists
                    resized.write_videofile(resizedOutputFile)
        n+=1
        
    video.reader.close()
    video.audio.reader.close_proc()

    if len(attackClips)>0:
        with open(os.path.join(base_dir, "processed", attacksFolder, 'attacks.txt'), 'w') as outfile:  
            json.dump(attackClips, outfile, indent=4)
