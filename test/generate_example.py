import json
import argparse

def read_from_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"Success to read file：{filename}")
        return loaded_data
    except Exception as e:
        print(f"Fail to read file：{str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser
    parser.add_argument('--path1', type=str, required=True)
    parser.add_argument('--path2', type=str, required=True)
    args = parser.parse_args()
    file1 = read_from_json(args.path1)
    file2 = read_from_json(args.path2)
    if "autoregressive" in args.path2:
        print("Compare to Autoregressive Inference:")
    elif "combine" in args.path2:
        print("Compare to Direct Combine:")
    result = []
    res_id = -1
    max_speed_up = 0
    for i in range(len(file1)):
        id = file1[i]["id"]
        c = next((item for item in file2 if item['id'] == id), None)
        s1 = file1[i]['length'] / (file1[i]['end_time'] - file1[i]['start_time'])
        s2 = c['length'] / (c['end_time'] - c['start_time'])
        speed_up = s1 / s2
        result.append([id, speed_up])
        if speed_up > max_speed_up:
            max_speed_up = speed_up
            res_id = id
    result.sort(key=lambda x: x[1], reverse=True)
    print(f"\nres_id={res_id}, speed_up={max_speed_up:.2f}x")
    print(f"count = {len(result)}")
    total_speedup = 0
    for r in result:
        total_speedup += r[1]
        print(f"ID: {r[0]}, Speed Up: {r[1]:.2f}x")
    print(f"\navg speedup: {total_speedup / len(result):.2f}x")


if __name__ == "__main__":
    main()