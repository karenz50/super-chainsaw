# assumes results are in this form 
# segments = [
#     {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00", "text": "Hello, how are you?"},
#     {"start": 4.5, "end": 7.2, "speaker": "SPEAKER_01", "text": "I'm good, thanks!"},
#     {"start": 7.2, "end": 10.0, "speaker": "SPEAKER_00", "text": "Great to hear that."}
# ]

import plotly.express as px
import pandas as pd
import pathlib
from datetime import datetime, timedelta

audio_path = pathlib.Path("audio.wav")

def run_test(segments):
    # convert segments into DataFrame
    data = []
    for seg in segments:
        data.append({
            "Speaker": seg["speaker"],
            "Start": seg["start"],
            "End": seg["end"],
            "Duration": seg["end"] - seg["start"],
            "Text": seg.get("text", "")
        })

    df = pd.DataFrame(data)

    base_time = datetime(1970, 1, 1)
    df["Start"] = df["Start"].apply(lambda x: base_time + timedelta(seconds=x))
    df["End"] = df["End"].apply(lambda x: base_time + timedelta(seconds=x))

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Speaker",
        color="Speaker",
        hover_data=["Text"]
    )
    fig.update_layout(
        title="Speaker Diarization Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Speaker",
        showlegend=True,
        hovermode="closest",
        height=400
    )

    # Add empty shape for playhead (vertical line)
    fig.update_layout(shapes=[{
        "type": "line",
        "x0": base_time,
        "x1": base_time,
        "y0": 0,
        "y1": 1,
        "xref": "x",
        "yref": "paper",
        "line": {
            "color": "black",
            "width": 2,
            "dash": "dot"
        }
    }])

    # save plot to HTML string
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id="plot")

    # add audio player and playhead JS logic
    audio_html = f"""
    <h2>Audio Player</h2>
    <audio id="audio" controls style="width: 100%;">
        <source src="{audio_path.name}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <br><br>

    <script>
    const audio = document.getElementById('audio');
    const baseTime = new Date(0);  // Jan 1, 1970

    audio.ontimeupdate = function() {{
        const currentTime = audio.currentTime * 1000;  // ms
        const currentDate = new Date(baseTime.getTime() + currentTime).toISOString();

        Plotly.relayout('plot', {{
            'shapes[0].x0': currentDate,
            'shapes[0].x1': currentDate
        }});
    }};
    </script>
    """

    # combine and write to final HTML
    output_html = f"""
    <html>
    <head>
        <title>Diarization Viewer</title>
        <meta charset="utf-8" />
    </head>
    <body>
    {audio_html}
    {plot_html}
    </body>
    </html>
    """

    output_path = pathlib.Path("timeline.html")
    output_path.write_text(output_html)

    print(f"open '{output_path}' in browser")
