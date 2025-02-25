v17_2
相较于v17_1的改动


结合双模态进行重建，相当于stage2 pretrain
使用17_1,stage1然后使用cross-attn进行结合预测(fix encoder)，decoder用MLP
