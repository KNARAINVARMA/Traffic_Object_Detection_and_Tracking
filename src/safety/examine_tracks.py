import pandas as pd

df = pd.read_csv(r"D:\btp\narain_data\test1.csv")

t14 = df[df["track_id"] == 14].sort_values("frame")
t19 = df[df["track_id"] == 19].sort_values("frame")

print("Track 14 (Shortcut West->South):")
print(t14[["frame", "world_x", "world_y"]].head(3))
print(t14[["frame", "world_x", "world_y"]].tail(3))

print("\nTrack 19 (Proper North->West):")
print(t19[["frame", "world_x", "world_y"]].head(3))
print(t19[["frame", "world_x", "world_y"]].tail(3))
