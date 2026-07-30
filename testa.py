import zlib
import base64

def compress_url(url):
    compressed = zlib.compress(url.encode('utf-8'), level=9)
    # Base85 is ~25% more efficient than Base64
    encoded = base64.b85encode(compressed).decode('utf-8')
    return encoded



if __name__ == '__main__':
    
    url = "https://bedrock-agentcore-runtime-060795913828-eu-central-1-8rqtu4h541.s3.eu-central-1.amazonaws.com/1000282833.png?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEM3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIH9yLJkVl9sqX%2Fl3r84L5Nk6IATUVGRC82KSa2Zuj6ScAiEAmfSJRbDhZIvG8UrjP6osDp2s6rKQ9RzywTEgSjw34VYqkQQIlv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwwNjA3OTU5MTM4MjgiDO2uL3D%2FYlpikPKQmirlA09ncgSazFHwkPsEf134PAHdvXeOmdL7Yn53fpzFmLsPDIIMT1SEQi2vvrYWKai1cLfEKb6gb4944Ts2eHiJopAQRV88HqhmwNt81vvo3rRz%2BZAmRX4VSqivFNcFIapFX0oRW99wek95zXcfjfVG6gD4I%2BteS8cvU7UMA8cCBRe65IcrIEaj7YV4b4xiP32Z1BTCiohR7JyinN%2BPzsXZqWObCnOu6LReMYjtVIU6K3zjL7jYoUNYf9J0s2JHPRl2ssObOV9r9upOXKiO1GchSueHNAylrcU0wwf8qAGHW1C0hTOMxyv%2BJpMTgHUrUA2zRuUJ2al%2FdW4WcQqYXWSqIx5yuOjQUIQrz7nQ4CfjCUIYTB6T%2FBpid5tznPCnkGxzA6V6SF3QoNiY9etj%2Ft02H%2FonbM%2F24KfQX2CXwA5SfUd01OxXaluSBo6SsmmRdfr03Y0uVTMdRacDCpegHo1LPf6%2F688kHDIVnzt2xKbQiFWDb%2FXArMcj6PWE28BT33XSuD0n1PO5uJefYqvLGAYUVVyPgBlO5rjnDJV6HLOIrNnoGf67r2JcEA3sftUCb1632JLa49g%2Bj1fkRUiCAc5VtL7S08tuDpqPMwYE91SvQUCivo18%2F%2FYobIWqvLN7TQGJa9paVulFMPyFrdMGOrsC9A0%2F462ag4EBpSNYIQ6aaY5TyavzYcyIroMJZ4DhWz%2FX6F0%2FGXKA8p%2Br0nlEn6uyEufB9PvugeODittXC2OM3BQpD6aEh0UW1zwHVUD4%2Fa4fEP8nk5cACtp0yEPhk3bpQaU5HT70Jn499p%2BKogpmpRP60dcwyOKHrxvCpLW4oXW2OxBBUqmumpHSOv6eyHHPFhs45dUBYacyn7xYBSUCJ55YxMUITbXN9wYHOQp6JCDnwgldo8BvKQ%2BkC8ooySgnEHQpeTXEHwiF0ykfe2vYKxTVJddXAun8q86SYvNFxLjztpf5C8HlwORj8eWQttbicWJm4w8t51O7Y779uJjxT111tq7B2JUaYgmTNRbDG6V%2F%2FnQX5b87LkiZEjNg%2BaxjV92iACm7T5sTovlmoVzypCaG0AWEgouLt0PL&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ4J5YAZSBFXE37CL%2F20260730%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260730T125015Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=d49a04a915c5fd9faea816bc507efb0e874d5d48498d2fc76cc3ff105665a950"
    
    print(compress_url(url))