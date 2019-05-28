### tensorflow keras 项目模板
* 将project重命名
	rename project into project name

* 修改setup.py里面项目名字，模板即可使用
	modify setup.py project name and template will works fine

此模板由[Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)修改而来 
Template modified with project: [Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)

1. setup.py配置项目依赖

    Use setup.py to configure project required packages
2. 新增data_generator以更新数据输入

    Add data_generator class for the Inputs
3. 新增model以输入网络设计

    Add model class for Network design
4. 复制config/run_train.yaml以创建配置

    make a version of project/config/run_train.yaml and configure the yaml file for training/other usage

5. python train.py -c my_run_train.yaml
