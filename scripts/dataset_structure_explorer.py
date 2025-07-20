import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
import json

def dataset2path(dataset_name):
    """Convert dataset name to GCS path."""
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"

def explore_tensor_structure(tensor, prefix="", max_depth=5, current_depth=0):
    """递归探索张量的结构"""
    if current_depth >= max_depth:
        return {"max_depth_reached": True}
    
    if isinstance(tensor, dict):
        result = {}
        for k, v in tensor.items():
            result[k] = explore_tensor_structure(v, prefix + "  ", max_depth, current_depth + 1)
        return result
    elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
        return {
            "type": f"{type(tensor).__name__}",
            "length": len(tensor),
            "sample": explore_tensor_structure(tensor[0], prefix + "  ", max_depth, current_depth + 1)
        }
    elif isinstance(tensor, tf.Tensor):
        return {
            "type": "tf.Tensor",
            "shape": tensor.shape.as_list(),
            "dtype": tensor.dtype.name
        }
    else:
        return {"type": f"{type(tensor).__name__}"}

def main():
    parser = argparse.ArgumentParser(description="探索Open-X-Embodiment数据集的结构")
    parser.add_argument("--dataset", type=str, default="berkeley_autolab_ur5", help="数据集名称")
    parser.add_argument("--episode", type=int, default=1, help="要加载的episode编号")
    args = parser.parse_args()
    
    print(f"正在加载数据集: {args.dataset}, episode: {args.episode}")
    
    # 创建一个结果JSON结构体
    result_json = {
        "dataset_name": args.dataset,
        "episode_number": args.episode,
        "episode_structure": {}
    }
    
    # 加载数据集
    try:
        b = tfds.builder_from_directory(builder_dir=dataset2path(args.dataset))
        ds = b.as_dataset(split=f"train[{args.episode}:{args.episode + 1}]")
        episode = next(iter(ds))
        
        print(f"成功加载数据集: {args.dataset}")
        result_json["load_status"] = "success"
        
        # 检查数据集结构
        if "steps" in episode:
            steps_count = len(episode['steps'])
            print(f"Episode中的步骤数: {steps_count}")
            result_json["steps_count"] = steps_count
            
            # 探索整个episode的结构
            print("\n===== Episode的顶级键 =====")
            top_level_keys = list(episode.keys())
            for key in top_level_keys:
                print(f"- {key}")
            result_json["top_level_keys"] = top_level_keys
            
            # 探索steps的结构 - 修改这部分代码以处理不同类型的steps
            print("\n===== Steps的结构 =====")
            steps = episode["steps"]
            steps_type = str(type(steps))
            print(f"Steps的类型: {steps_type}")
            result_json["steps_type"] = steps_type
            
            # 根据steps的类型采取不同的处理方式
            if isinstance(steps, (list, tuple)) and len(steps) > 0:
                # 如果steps是列表或元组，直接访问第一个元素
                print("\n===== 第一个Step的顶级键 =====")
                step = steps[0]
                step_keys = list(step.keys())
                for key in step_keys:
                    print(f"- {key}")
                result_json["first_step_keys"] = step_keys
                
                # 详细探索第一个step的结构
                print("\n===== 第一个Step的详细结构 =====")
                structure = explore_tensor_structure(step)
                print(json.dumps(structure, indent=2))
                result_json["first_step_structure"] = structure
                
            elif hasattr(steps, "take") and callable(getattr(steps, "take", None)):
                # 如果steps是一个数据集对象，使用take方法获取第一个元素
                print("\n===== Steps是一个数据集对象 =====")
                result_json["steps_is_dataset"] = True
                try:
                    first_step = next(iter(steps.take(1)))
                    print("\n===== 第一个Step的顶级键 =====")
                    step_keys = list(first_step.keys())
                    for key in step_keys:
                        print(f"- {key}")
                    result_json["first_step_keys"] = step_keys
                    
                    # 详细探索第一个step的结构
                    print("\n===== 第一个Step的详细结构 =====")
                    structure = explore_tensor_structure(first_step)
                    print(json.dumps(structure, indent=2))
                    result_json["first_step_structure"] = structure
                except Exception as e:
                    error_msg = str(e)
                    print(f"无法访问第一个step: {error_msg}")
                    result_json["first_step_error"] = error_msg
                    
                    print("尝试打印steps的基本信息:")
                    steps_info = {"type": str(type(steps))}
                    print(f"  - 类型: {steps_info['type']}")
                    
                    if hasattr(steps, "element_spec"):
                        element_spec = str(steps.element_spec)
                        print(f"  - 元素规格: {element_spec}")
                        steps_info["element_spec"] = element_spec
                    
                    result_json["steps_info"] = steps_info
            else:
                # 如果steps是其他类型，尝试打印其基本信息
                print(f"Steps是一个不支持直接索引的类型: {type(steps)}")
                result_json["steps_indexable"] = False
                
                print("尝试打印steps的基本信息:")
                steps_info = {}
                if hasattr(steps, "__dict__"):
                    for attr_name in dir(steps):
                        if not attr_name.startswith("_") and not callable(getattr(steps, attr_name)):
                            try:
                                attr_value = getattr(steps, attr_name)
                                attr_value_str = str(attr_value)
                                print(f"  - {attr_name}: {attr_value_str}")
                                steps_info[attr_name] = attr_value_str
                            except:
                                pass
                result_json["steps_info"] = steps_info
            
            # 保存完整的episode结构
            result_json["episode_structure"] = explore_tensor_structure(episode)
            
            # 保存结果到文件
            output_file = f"{args.dataset}_structure.json"
            with open(output_file, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"\n结构已保存到文件: {output_file}")
            
        else:
            print("警告: 数据集中没有'steps'键")
            result_json["warning"] = "数据集中没有'steps'键"
            
            print("可用的顶级键:")
            top_level_keys = list(episode.keys())
            for key in top_level_keys:
                print(f"- {key}")
            result_json["top_level_keys"] = top_level_keys
            
            # 保存结果到文件
            output_file = f"{args.dataset}_structure.json"
            with open(output_file, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"\n结构已保存到文件: {output_file}")
    
    except Exception as e:
        error_msg = str(e)
        print(f"错误: {error_msg}")
        result_json["error"] = error_msg
        
        # 即使出错也保存已收集的信息
        output_file = f"{args.dataset}_structure.json"
        with open(output_file, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"\n部分结构已保存到文件: {output_file}")

if __name__ == "__main__":
    main()