# create_clips.py

This script can be used to cutting out clips with shots and no-shots + attacks (shots nearby the goal area or in other parts of the fields, to be used as negative examples) and putting them into `source_dir`.

### Worklow

1. It loads matches to process from `matches.json` file
2. It filters matches according to a list of `matchIds` from `skip.json`.
3. For each match, takes some metadata and parameters and process it to find specific 'shot'/'goal' events in the annotation file (`Marks.jsonl`). For each of the event, cut a 'shot' subclip,  a 'no-shot' subclip in the middle of the previous clip and the current one. It also produce a number of 'attacks' not overlapping with any 'shot' events.

----

### Debug / Clip validation mode

A number of command line options are useful for debugging purposes, parameters check, or produced clip validation.

To debug/process a single specific match, with various options:

    python .\create_clips.py --debug True --debugMaxClips 2 --debugMatchId 123456 --debugShotsOnly True --overwriteExisting False

This will process match 123456, extracting max 2 clips, only shots (no no-shots and attacks); if clips already exist, don't overwrite them.

General Usage:

    create_clips.py [-h] [--debugMatchId DEBUGMATCHID] [--debugMaxClips DEBUGMAXCLIPS] [--debug DEBUG] [--debugShotsOnly DEBUGSHOTSONLY] [--goalClipEnabled GOALCLIPENABLED] [--overwriteExisting OVERWRITEEXISTING]

optional arguments:

  **--debugMatchId [matchId]**  
    Produce clips for the specified match only

  **--debugMaxClips N**  
    Produce up to N clips for the specified match/half time only

  **--debug [True|False]**  
    Enable detailed traces for each clip

  **--debugShotsOnly [True|False]**  
    Produce Shots clip only (no no-shots | attacks)

  **--overwriteExisting [True|False]**  
    If a clip exists, overwrite it or not

  **--goalClipEnabled [True|False]**  
    Toggle "Goal" Shot clips production
