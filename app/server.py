from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai import *
from fastai.vision import *
import base64
import pdb
from utils import *

get_y_fn = lambda x: os.path.join(path_lbl, f'{x.stem}_groundtruth.png')

def iou(input, targs, iou=True, eps=1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input+targs).sum(dim=1).float()
    if not iou: l = 2. * intersect / union
    else: l = intersect / (union-intersect+eps)
    l[union == 0.] = 1.
    return l.mean()

export_file_url = 'https://www.dropbox.com/s/5x70oksyu3xyiqe/model.8.30?dl=1'
export_file_name = 'model.8.30.pkl'
classes = ['0','1']

path = Path(__file__).parent
path_lbl = path/'groundtruths'
templates = Jinja2Templates(directory='app/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    await download_file(export_file_url, path/'models'/export_file_name)
    defaults.device = torch.device('cpu')
    learn = load_learner(path/'models', export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())

    img = open_image(BytesIO(img_bytes))
    #x, y, z = img.data.shape
    #
    #max_size = 1000
    #y_new, z_new = get_resize(y, z, max_size)

    #data_bunch = (ImageImageList.from_folder(path).split_none().label_from_func(lambda x: x)
     #     .transform(get_transforms(do_flip=False), size=(y_new,z_new), tfm_y=True)
     #     .databunch(bs=2, no_check=True).normalize(imagenet_stats, do_y=True))

    #data_bunch.c = 2
    # Classes (i.e. the possible values in the mask .png)
    codes = ['0', '1']

    src = (SegmentationItemList.from_folder(path=path/'Dataset')
        .split_by_folder(train='train', valid='valid')
        .label_from_func((get_y_fn), classes=codes))

    ds_tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180., max_zoom=1.1,
                   max_lighting=0.13, max_warp=0, p_affine=0.75,
                   p_lighting=0.75)
    size = (224,224)
    bs = 32
    size = (224,224)
    bs = 32
    data = (src.transform(ds_tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
    
    #data = (src.transform(ds_tfms, size=size, tfm_y=True)
    #    .databunch(bs=bs)
    #   .normalize(imagenet_stats))
    
    learn.data = data
    _,img_hr,losses = learn.predict(img)

    im = Image(img_hr.clamp(0,1))

    im_data = image2np(im.data*255).astype(np.uint8)

    img_io = BytesIO()

    PIL.Image.fromarray(im_data).save(img_io, 'PNG')

    img_io.seek(0)

    img_str = base64.b64encode(img_io.getvalue()).decode()
    img_str = "data:image/png;base64," + img_str

    return templates.TemplateResponse('output.html', {'request' : request, 'b64val' : img_str})


@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
