<script lang="ts">
	
	import { actual_beer} from './store.js';
	let beer;
	actual_beer.subscribe
	(value => {beer = value})
		
	let beer_name="PLACE_HOLDER"
	let img = "./tmp.png"
	let beer_desc = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec gravida tortor eget felis molestie, non luctus quam aliquam. Nunc et arcu est. Nulla at lacus lorem. Mauris faucibus bibendum magna vel facilisis. Integer ut ligula quis sem efficitur luctus. Pellentesque a lacus nec magna mollis dapibus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam in nibh ac magna ultricies mattis. Sed elementum arcu et lacus sollicitudin, a finibus mauris faucibus. Mauris diam arcu, sollicitudin et tortor at, convallis euismod nunc. Donec tempus mauris metus, in tempor orci pharetra eget. Nam euismod massa non egestas lacinia. Ut rutrum quam at mauris interdum, nec dictum ex finibus. Maecenas sit amet congue ipsum. Sed volutpat nisi ut vehicula congue. Donec justo ante, elementum quis ullamcorper sit amet, venenatis vitae tellus."
	
</script>
	<div style="min-width: 100px;" class='content_holder'>
		{#if typeof beer !== "undefined"}
		<h1 style="color:#e6a14e">{beer[0]}</h1>
		<div class='img_con'>
			<img src={beer[3]} alt="focused beer" class="beer_img">
			<div class="rat">
				{beer[2]}
			</div>
		</div>
		
		<div class="content-scroller">
			<p style="">Browar:&nbsp;</p><p>{beer[1] ?? ""}</p>
			<p style="">Ocena:&nbsp;</p><p>{beer[2] ?? ""}</p>
			<p style="">ibu:&nbsp;</p><p>{beer[7]?? ""}</p>
			<p style="">STYL:&nbsp;</p><p>{beer[8] ?? ""}</p>
			<p style="">alc:&nbsp;</p><p>{beer[6]?? ""}</p>
			<p style="">Chmiele:&nbsp;</p><p>{beer[9] ?? ''}</p>
			<p style="">DROŻDŻE:&nbsp;</p><p>{beer[11]?? ""}</p>
			<p style="">KONTRAKTOR:&nbsp;</p><p>{beer[12]?? ""}</p>
			<p style="">STYL:&nbsp;</p><p>{beer[8] ?? ""}</p>
			<p style="">SŁODY:&nbsp;</p><p>{beer[10]?? ""}</p>
			<p style="">blg:&nbsp;</p><p>{beer[5]?? ""}</p>
			
<!-- 			
			TUTAJ POWINNIEN BYĆ WYKRES
			
			<img src={img} >  -->
		</div>
		{/if}	
	  </div>


<style>
	.img_con{
		position: relative;
		width: 100%;
	}
	.rat{
		text-align: right;
		position: absolute;

		bottom: 0;
		right: 0;
		color:#e6a14e;
		font-size: 3em;
		text-shadow: 2px 2px 4px #000000;
	}
	.content-scroller{
		display: flex;
		flex-wrap: wrap;
		overflow: scroll;
		overflow-x: hidden;
		height: 60%;
		width: 100%;
		margin: 0 0 0 0;
		
		
		
	}
	.beer_img{
		width: 100%;
	}
	.content_holder{
		display: flex;
		flex-direction: column;
		align-items: center;
		box-shadow: #e6a24e72 0px 8px 16px -12px inset, #e6a24e72 0px 8px 16px -18px inset;
		
		
	}

	.content_holder p:nth-child(odd){
		text-align: right;
		
	}
	
	.content_holder p:nth-child(4n+1), .content_holder p:nth-child(4n+2) {
		background-color: #e6a14e;
		color: #333

	}

	 .content_holder p {
		margin: 0;
		flex-basis: 50%;
		color:#e6a14e;
		background-color: #333
	}

	.content-scroller::-webkit-scrollbar {
  width: 0.5em;
}
 
.content-scroller::-webkit-scrollbar-track {
  box-shadow: inset 0 0 3px rgba(0, 0, 0, 0.3);
}
 
.content-scroller::-webkit-scrollbar-thumb {
  background-color: darkgrey;
  outline: 1px solid slategrey;
}
</style>