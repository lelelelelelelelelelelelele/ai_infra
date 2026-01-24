"""
弃用警告的pytest测试
"""
import os
import warnings
import shutil
import pytest
from ai_infra import init_ai_config
from ai_infra import _init_ai_config_fallback


def test_deprecation_warning():
    """测试弃用警告功能"""
    
    # 直接调用fallback函数来演示警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 调用会触发弃用警告的函数
        config = _init_ai_config_fallback("gpt")
        
        # 检查是否产生了弃用警告
        assert len(w) > 0, "应该捕获到弃用警告"
        for warning in w:
            # 验证警告类型和内容
            assert issubclass(warning.category, (DeprecationWarning, UserWarning, FutureWarning)), \
                f"警告类型应该是弃用相关类型，实际为: {warning.category.__name__}"
            assert str(warning.message), "警告消息不应为空"
                
        assert config is not None, "配置不应为None"


def test_yaml_config_warning():
    """测试YAML配置警告功能"""
    
    config_path = "ai_models.yaml"
    backup_path = "ai_models.yaml.bak"
    
    # 重命名配置文件以触发警告
    config_existed = os.path.exists(config_path)
    if config_existed:
        shutil.move(config_path, backup_path)
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 调用init_ai_config，由于没有配置文件会触发警告
            config = init_ai_config("gpt")
            
            # 检查是否产生了警告
            assert len(w) > 0, "应该捕获到配置文件缺失的警告"
            for warning in w:
                # 验证警告类型和内容
                assert issubclass(warning.category, (UserWarning, DeprecationWarning)), \
                    f"警告类型应该是UserWarning或相关弃用类型，实际为: {warning.category.__name__}"
                assert str(warning.message), "警告消息不应为空"
                
            assert config is not None, "配置不应为None"
            
    finally:
        # 恢复配置文件
        if config_existed and os.path.exists(backup_path):
            shutil.move(backup_path, config_path)


if __name__ == "__main__":
    pytest.main([__file__])
