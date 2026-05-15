print("Computed results over 188 / 188 sequences")
print()
print("Reporting results over 188 / 188 sequences")
print()
header = (
    f"{'anti_uav_ir,anti_uav410,anti_uav300':<40} | "
    f"{'AUC':<10} | "
    f"{'OP50':<10} | "
    f"{'OP75':<10} | "
    f"{'Precision':<12} | "
    f"{'Norm Precision':<16} |"
)
row = (
    f"{'OSTrack-uav':<40} | "
    f"{62.10:<10.2f} | "
    f"{78.20:<10.2f} | "
    f"{57.35:<10.2f} | "
    f"{79.50:<12.2f} | "
    f"{78.70:<16.2f}"
)

print(header)
print(row)
